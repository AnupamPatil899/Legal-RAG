import logfire

logfire.configure(service_name="test")

with logfire.span("test"):
    print("hello")