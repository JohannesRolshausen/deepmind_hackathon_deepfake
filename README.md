## How to run it

### Command Line Interface

```bash
python main.py --img "PATH_TO_IMG" --text "USER_TEXT"
```

### Web Frontend (Recommended for Hackathon Demo)

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the Flask server:

```bash
python app.py
```

3. Open your browser and navigate to:

```
http://localhost:5000
```

4. Upload an image and optionally provide context text, then click "Analyze Image"

The web interface provides:

- **Real-time progress updates** showing each tool's analysis
- **Final result display** with probability score and explanation
- **Individual tool outputs** for detailed inspection
- **Modern, visually appealing UI** perfect for hackathon demos

## Folder Structure

<img width="284" height="236" alt="image" src="https://github.com/user-attachments/assets/0488aac7-c0f4-4721-b8f4-fcb2100c9bbc" />

## How to add a step

1. pull main.py
2. put your step into /steps
3. make sure your step implements the BaseStep interface: Have a method "run" that takes a TaskInput and returns a StepResult (see /steps/exampletoolx.py for an example)
4. Tell Johannes. He will update the main.py file to include your step in the pipeline. (Needed to prevent merge conflicts)

## API KEY

1. please add your api key to an .env file that is only on your local machine. It should look like this:

```
GEMINI_API_KEY=your_api_key_here
```
