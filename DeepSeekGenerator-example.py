from haystack import Pipeline

# Create an instance of your custom generator
deepseek_generator = DeepSeekGenerator(
    api_key="Your API KEY",
    model="deepseek-chat",
    temperature=1.0,
    max_tokens=1024
)

# Create a pipeline and add the generator
pipeline = Pipeline()
pipeline.add_component("generator", deepseek_generator)

# Run the pipeline with a query and optional system role, if not set system role it will fall back to default 
system_role = "You are an expert in geography."
query = "What is the capital of France?"
result = pipeline.run({"generator": {"query": query, "system_role": system_role}})

# Print the generated response
print(result["generator"]["response"])
