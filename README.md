# acquring huggingface API key

* Go to https://huggingface.co/ and sign up for an account.

* Go to https://huggingface.co/settings/tokens and generate a new token with write access.

* Copy the token and paste in the main.py here:

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YOUR HUGGINGFACE API TOKEN"

# installing the required libraries

run in terminal:

```
pip install -r requirements.txt
```

# running the code

run in terminal:

```
python main.py
```