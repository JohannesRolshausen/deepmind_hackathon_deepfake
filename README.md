## How to run it

run:
```python main.py --img "PATH_TO_IMG" --text "USER_TEXT"```


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