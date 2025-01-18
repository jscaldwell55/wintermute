from dotenv import load_dotenv
import os

load_dotenv()  # This should not raise any errors
print(os.getenv("INDEX_NAME"))  # Should print "wintermute" if .env is configured correctly
