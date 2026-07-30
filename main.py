import logfire

logfire.configure()
logfire.instrument_system_metrics()
logfire.info("Hello, {place}!", place="World")
