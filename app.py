import json
import os
import queue
import threading
from typing import Callable, List, Optional

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

from core.llm import call_llm
from core.schemas import AggregatedContext, TaskInput
from steps.ai_metadata_analyzer import AIMetadataAnalyzer
from steps.base import BaseStep
from steps.judge_system import JudgeSystem
from steps.reverse_image_search import ReverseImageSearch
from steps.synthid_detection import SynthIDDetection
from steps.visual_forensics import VisualForensicsAgent

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

# Global queue for progress updates
progress_queue = queue.Queue()

def send_progress_update(event_type: str, data: dict):
    """Send a progress update to the frontend"""
    progress_queue.put({
        'type': event_type,
        'data': data
    })

def run_pipeline(image_path: str, user_text: str = ""):
    """Run the deepfake detection pipeline with progress callbacks"""
    try:
        # Setup Context & Input
        task_input = TaskInput(image_path=image_path, text=user_text)
        context = AggregatedContext(task_input=task_input)
        
        # Register Steps
        steps: List[BaseStep] = [
            ReverseImageSearch(),
            SynthIDDetection(),
            VisualForensicsAgent(),
            JudgeSystem(),
            AIMetadataAnalyzer()
        ]
        
        # Map step names to display names
        step_names = {
            'ReverseImageSearch': 'Reverse Image Search',
            'SynthIDDetection': 'SynthID Detection',
            'VisualForensicsAgent': 'Visual Forensics',
            'JudgeSystem': 'Judge System Debate',
            'AIMetadataAnalyzer': 'Metadata Analysis'
        }
        
        send_progress_update('start', {
            'message': 'Pipeline started',
            'total_steps': len(steps)
        })
        
        # Sequential execution with progress updates
        for idx, step in enumerate(steps):
            step_name = step.__class__.__name__
            display_name = step_names.get(step_name, step_name)
            
            send_progress_update('step_start', {
                'step': step_name,
                'display_name': display_name,
                'step_number': idx + 1,
                'total_steps': len(steps)
            })
            
            try:
                result = step.run(task_input)
                context.results.append(result)
                
                # Format result for frontend
                result_data = {
                    'source': result.source,
                    'content': result.content
                }
                
                send_progress_update('step_complete', {
                    'step': step_name,
                    'display_name': display_name,
                    'result': result_data
                })
                
            except Exception as e:
                error_msg = str(e)
                send_progress_update('step_error', {
                    'step': step_name,
                    'display_name': display_name,
                    'error': error_msg
                })
                print(f"❌ error in step {step_name}: {e}")
        
        # Final LLM Call
        send_progress_update('final_analysis_start', {
            'message': 'Performing final analysis...'
        })
        
        final_answer = call_llm(context)
        
        try:
            # Clean up JSON response - handle markdown code blocks
            cleaned_answer = final_answer.strip()
            if cleaned_answer.startswith('```'):
                # Remove markdown code block markers
                lines = cleaned_answer.split('\n')
                # Remove first line if it's ```json or ```
                if lines[0].startswith('```'):
                    lines = lines[1:]
                # Remove last line if it's ```
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                cleaned_answer = '\n'.join(lines).strip()
            
            data = json.loads(cleaned_answer)
            
            send_progress_update('final_result', {
                'probability_score': data.get('probability_score'),
                'explanation': data.get('explanation'),
                'full_context': context.model_dump()
            })
            
        except json.JSONDecodeError as e:
            # Try to extract JSON from the response if it's embedded in text
            import re
            json_match = re.search(r'\{[^{}]*"probability_score"[^{}]*\}', cleaned_answer, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    send_progress_update('final_result', {
                        'probability_score': data.get('probability_score'),
                        'explanation': data.get('explanation'),
                        'full_context': context.model_dump()
                    })
                except:
                    send_progress_update('final_result', {
                        'probability_score': None,
                        'explanation': final_answer,
                        'raw_output': True,
                        'full_context': context.model_dump()
                    })
            else:
                send_progress_update('final_result', {
                    'probability_score': None,
                    'explanation': final_answer,
                    'raw_output': True,
                    'full_context': context.model_dump()
                })
        
        send_progress_update('complete', {
            'message': 'Pipeline completed successfully'
        })
        
    except Exception as e:
        send_progress_update('error', {
            'error': str(e)
        })

@app.route('/')
def index():
    """Serve the frontend"""
    return send_from_directory('frontend', 'index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Start the analysis pipeline"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    user_text = request.form.get('text', '')
    
    if file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400
    
    # Save uploaded file temporarily
    upload_dir = 'uploads'
    os.makedirs(upload_dir, exist_ok=True)
    
    filepath = os.path.join(upload_dir, file.filename)
    file.save(filepath)
    
    # Clear the progress queue
    while not progress_queue.empty():
        try:
            progress_queue.get_nowait()
        except queue.Empty:
            break
    
    # Start pipeline in background thread
    thread = threading.Thread(target=run_pipeline, args=(filepath, user_text))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'status': 'started',
        'message': 'Analysis started'
    })

@app.route('/api/progress')
def progress():
    """Server-Sent Events endpoint for real-time progress updates"""
    def generate():
        while True:
            try:
                # Wait for progress update (with timeout)
                update = progress_queue.get(timeout=1)
                yield f"data: {json.dumps(update)}\n\n"
            except queue.Empty:
                # Send heartbeat to keep connection alive
                yield ": heartbeat\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/upload-local', methods=['POST'])
def upload_local():
    """Handle local file path upload"""
    data = request.json
    image_path = data.get('image_path')
    user_text = data.get('text', '')
    
    if not image_path or not os.path.exists(image_path):
        return jsonify({'error': 'Invalid image path'}), 400
    
    # Clear the progress queue
    while not progress_queue.empty():
        try:
            progress_queue.get_nowait()
        except queue.Empty:
            break
    
    # Start pipeline in background thread
    thread = threading.Thread(target=run_pipeline, args=(image_path, user_text))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'status': 'started',
        'message': 'Analysis started'
    })

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, port=5000, threaded=True)

