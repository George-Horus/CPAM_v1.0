"""
Module: project_structure_exporter

This script automatically generates a directory structure tree from a Python project folder and can output:
    - JSON format project structure data
    - XMind Zen format mind map file (.xmind)

Main Features:
1. Traverses the project folder, ignoring common irrelevant directories (e.g., .git, __pycache__, etc.)
2. Parses each Python file (.py) to extract:
    - Top-level functions
    - Top-level classes and methods defined within them
    - The first-line docstring description of each function/method
3. Saves the above information as a hierarchical structure, suitable for:
    - Saving to JSON files
    - Exporting as an XMind Zen mind map

Main Functions:
- build_topic(path)
    Generates the project directory tree starting from the specified root path.
- parse_python_defs(filepath)
    Parses a single Python file for function, class, and method definitions.
- save_json(data, filename)
    Saves the structure tree as a JSON file.
- save_xmindzen(root_data, filename)
    Exports the structure tree as an XMind Zen (.xmind) file.

Input:
- Project root path
- Output file name (JSON or XMind)

Output:
- JSON file (project structure tree)
- XMind file (for visualizing project structure)
"""

import os
import ast
import json
import uuid
import zipfile
from datetime import datetime

IGNORE_DIRS = {'.idea', '.venv', '__pycache__', '.git'}

def parse_python_defs(filepath):
    """
    Parses a Python file to extract top-level classes, functions,
    and methods defined within classes, including the first-line docstring.

    Args:
        filepath (str): Path to the Python file to be parsed.

    Returns:
        list: A list of dictionaries, each representing a class/function/method node
              in the format {"title": "def xxx() 📘 doc"}.
    """
    def_node_list = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=filepath)

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                title = f"def {node.name}()"
                doc = ast.get_docstring(node)
                if doc:
                    doc_first_line = doc.strip().split('\n')[0]
                    title += f' 📘 "{doc_first_line}"'
                def_node_list.append({"title": title})

            elif isinstance(node, ast.ClassDef):
                class_title = f"class {node.name}()"
                class_doc = ast.get_docstring(node)
                if class_doc:
                    class_doc_first_line = class_doc.strip().split('\n')[0]
                    class_title += f' 📘 "{class_doc_first_line}"'
                class_children = []

                for subnode in node.body:
                    if isinstance(subnode, ast.FunctionDef):
                        method_title = f"def {subnode.name}()"
                        method_doc = ast.get_docstring(subnode)
                        if method_doc:
                            method_doc_first_line = method_doc.strip().split('\n')[0]
                            method_title += f' 📘 "{method_doc_first_line}"'
                        class_children.append({"title": method_title})

                class_node = {
                    "title": class_title,
                    "children": {"attached": class_children} if class_children else {}
                }
                def_node_list.append(class_node)

    except Exception as e:
        def_node_list.append({"title": f"[Error parsing: {e}]"})
    return def_node_list

def build_topic(path):
    """
    Builds a project structure tree from the given path,
    including folders, Python files, and class/function definitions.

    Args:
        path (str): Root path of the project folder to traverse.

    Returns:
        dict: A nested dictionary tree containing `title` and `children` fields,
              suitable for mind map output.
    """
    name = os.path.basename(path)
    node = {"title": name, "children": {"attached": []}}

    try:
        entries = sorted(os.listdir(path))
    except PermissionError:
        return node

    entries = [e for e in entries if e not in IGNORE_DIRS]
    for entry in entries:
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            node["children"]["attached"].append(build_topic(full_path))
        elif entry.endswith(".py"):
            py_node = {
                "title": entry,
                "children": {"attached": parse_python_defs(full_path)}
            }
            node["children"]["attached"].append(py_node)
    return node

def save_json(data, filename="project_structure.json"):
    """
    Saves the project structure tree as a JSON file
    for further processing or display.

    Args:
        data (dict): Project structure dictionary returned by build_topic.
        filename (str): Output JSON file path.
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON export successful: {filename}")

def save_xmindzen(root_data, filename="project_structure.xmind"):
    """
    Exports the project structure as an XMind Zen file
    (.xmind, used by XMind 2020+).

    Args:
        root_data (dict): Project structure tree returned by build_topic.
        filename (str): Output XMind file path (.xmind).
    """
    topic_id = str(uuid.uuid4())
    content = [{
        "id": topic_id,
        "title": "Project Structure",
        "rootTopic": root_data,
        "format": "node_tree"
    }]

    manifest = {
        "file-entries": {
            "content.json": {"media-type": "application/json"},
            "metadata.json": {"media-type": "application/json"},
            "manifest.json": {"media-type": "application/json"},
        }
    }

    metadata = {
        "creator": "ChatGPT Python Script",
        "created": datetime.utcnow().isoformat() + "Z",
        "xmind-version": "2020",
        "license": "CC BY-SA 4.0"
    }

    with zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("content.json", json.dumps(content, ensure_ascii=False, indent=2))
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        zf.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))

    print(f"✅ XMind Zen export successful: {filename}")

if __name__ == "__main__":
    # Modify to your project path
    project_path = r"D:\Code_Store\CPAM_v1.0"
    structure_data = build_topic(project_path)

    # Export JSON + XMind
    save_json(structure_data)
    save_xmindzen(structure_data)
