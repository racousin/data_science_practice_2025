import jinja2

# Define the template environment with the FileSystemLoader
templateLoader = jinja2.FileSystemLoader(searchpath="./")
templateEnv = jinja2.Environment(loader=templateLoader)

# Load the template from file
template = templateEnv.get_template("template.jinja")

# Data to render the template with
modules_data = [
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
]  # Example data, replace with [1, 2, 2, 5, 6, 7, 1, 4] or other configurations

# Render the template with data
output = template.render(modules=modules_data)

# Print or save the output
print(output)
