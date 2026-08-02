import os

import logfire
from dotenv import load_dotenv

load_dotenv("../.env", override=True)

logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.info("Hello, {place}!", place="World")

# {
#   "token": "pylf_v1_us_khGSgXm3zzxddwnqjnVLwDMfSSmCV266fqXfTZ5cZMd5",
#   "project_name": "legal-rag",
#   "project_url": "https://logfire-us.pydantic.dev/yashwantpatil6403/legal-rag",
#   "logfire_api_url": "https://logfire-us.pydantic.dev"
# }
