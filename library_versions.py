import ast
import pkg_resources
import sys
import re

def get_imported_libraries(file_path):
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read(), filename=file_path)

    imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom)]
    
    libraries = set()
    for imp in imports:
        if isinstance(imp, ast.Import):
            for alias in imp.names:
                libraries.add(alias.name.split('.')[0])
        elif isinstance(imp, ast.ImportFrom):
            module = imp.module.split('.')[0]
            libraries.add(module)

    return libraries

def get_library_versions(libraries):
    # Mapping of import names to package names
    name_mapping = {'PIL': 'Pillow',
                    'sklearn': 'scikit-learn'
                    }

    found_versions = {}
    not_installed = []
    for library in libraries:
        package_name = name_mapping.get(library, library)
        try:
            version = pkg_resources.get_distribution(package_name).version
            found_versions[library] = version
        except pkg_resources.DistributionNotFound:
            not_installed.append(library)
    return found_versions, not_installed

def write_to_readme(readme_path, found_versions):
    with open(readme_path, 'r') as readme_file:
        readme_content = readme_file.read()

    # Check if the Libraries section exists
    if '## Libraries' in readme_content:
        # Replace the existing content with the updated library versions
        pattern = r'## Libraries.*?## Found Library Versions\n'
        replacement = f'### Libraries\n\n### Found Library Versions\n'
        updated_content = re.sub(pattern, replacement, readme_content, flags=re.DOTALL)
    else:
        # If the section does not exist, just append to the end of the file
        updated_content = readme_content + '\n### Found Library Versions\n'

    # Append the library versions
    with open(readme_path, 'w') as readme_file:
        readme_file.write(updated_content)
        for library, version in found_versions.items():
            readme_file.write(f"- {library}: {version}\n")


if __name__ == "__main__":
    
    if len(sys.argv) == 3:

        python_file_path = sys.argv[1]
        readme_path = sys.argv[2]

    else:
        AttributeError("Please provide a python file path as an argument and a readme path as an argument")
    
    imported_libraries = get_imported_libraries(python_file_path)
    found_versions, not_installed = get_library_versions(imported_libraries)

    print("Versions found:")
    for library, version in found_versions.items():
        print(f"{library}: {version}")

    print("\nNot installed:")
    for library in not_installed:
        print(library)
        
    write_to_readme(readme_path, found_versions)