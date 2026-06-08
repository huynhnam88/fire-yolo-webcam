from dotenv import load_dotenv
import os

load_dotenv()

print(os.getenv("BOT_TOKEN"))
print(os.getenv("CHAT_ID"))
print(os.getenv("CAMERA_URL"))