from utils.services.api_service import ApiService

api_service = ApiService()
for chunk in api_service.generate("Hello, what is your name?"):
    print(chunk)
