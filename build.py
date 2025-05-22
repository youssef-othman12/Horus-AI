import os
import shutil
from jinja2 import Environment, FileSystemLoader

# Create dist directory
if os.path.exists('dist'):
    shutil.rmtree('dist')
os.makedirs('dist')

# Copy static files
if os.path.exists('static'):
    shutil.copytree('static', 'dist/static')

# Setup Jinja2 environment
env = Environment(loader=FileSystemLoader('templates'))

# Convert Flask templates to static HTML
templates = {
    'horos1.html': 'index.html',
    'about_us.html': 'about_us.html',
    'page2_image_result.html': 'page2_image_result.html',
    'page3_recommendation_result.html': 'page3_recommendation_result.html'
}

for template_name, output_name in templates.items():
    template = env.get_template(template_name)
    output = template.render()
    
    with open(f'dist/{output_name}', 'w', encoding='utf-8') as f:
        f.write(output)

print("Static site generated successfully!")