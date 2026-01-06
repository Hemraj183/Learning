import os
import json

COURSE_MAT = "course_materials"

def add_link_to_notebooks():
    for week_folder in os.listdir(COURSE_MAT):
        folder_path = os.path.join(COURSE_MAT, week_folder)
        if not os.path.isdir(folder_path): continue
        
        # Look for notebook
        nb_path = os.path.join(folder_path, "notebook.ipynb")
        if os.path.exists(nb_path):
            print(f"Linking {week_folder}...")
            
            with open(nb_path, "r") as f:
                data = json.load(f)
            
            # Create Link Cell
            link_url = f"../../interactive_platform/modules/{week_folder}/interactive.html"
            
            # Markdown cell
            link_cell = {
                "cell_type": "markdown",
                "metadata": {
                    "alert": "info" # Optional styling
                },
                "source": [
                    f"# 🚀 Interactive Mode Available!\n",
                    f"\n",
                    f"Typical static notebooks are boring. We have a dedicated interactive module for this week.\n",
                    f"\n",
                    f"[👉 **Click here to open the Interactive Visualization**]({link_url})\n",
                    f"\n",
                    f"*(Note: Open this link in a new tab to keep the notebook running)*"
                ]
            }
            
            # Insert at top (index 0)
            data["cells"].insert(0, link_cell)
            
            with open(nb_path, "w") as f:
                json.dump(data, f, indent=1)
                
if __name__ == "__main__":
    add_link_to_notebooks()
    print("Notebooks linked successfully.")
