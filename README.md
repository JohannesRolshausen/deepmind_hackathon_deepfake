# Folder Structure

deepmind_hackathon_deepfake/
│
├── .env                # API Keys etc.
├── main.py             # Entry Point
│
├── core/
│   ├── llm.py          # LLM Wrapper
│   └── schemas.py      # Pydantic Models ( DTOs/POJOs)
│
└── steps/            # the actual intereseting steps
    ├── __init__.py
    ├── base.py         # Interface Definition
    ├── tool1.py  
    └── tool2.py ...
