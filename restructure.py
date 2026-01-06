import os
import shutil
import pathlib

# Configuration
BASE_DIR = os.getcwd()
COURSE_MAT = os.path.join(BASE_DIR, "course_materials")
INTERACTIVE_PLAT = os.path.join(BASE_DIR, "interactive_platform")
MODULES_DIR = os.path.join(INTERACTIVE_PLAT, "modules")

# Extensions
CODE_EXT = ['.py', '.ipynb', '.md']
WEB_EXT = ['.html', '.css', '.js']

# Mapping of Old Paths to New Week Names (Standardized)
# Format: (Old_Parent_Folder, Old_Week_Folder_Name) -> New_Week_Basename
WEEK_MAPPINGS = {
    # Week 1
    ("pytorch-tutorial", None): "week1_pytorch", 
    
    # Week 2-4 (llm_foundations)
    ("llm_foundations", "week2_transformer"): "week2_transformer",
    ("llm_foundations", "week3_llm_variants"): "week3_llm_variants",
    ("llm_foundations", "week4_router"): "week4_router",
    
    # Week 5-8 (diffusion_models)
    ("diffusion_models", "week5_diffusion"): "week5_diffusion",
    ("diffusion_models", "week6_unet"): "week6_unet",
    ("diffusion_models", "week7_ldm"): "week7_ldm",
    ("diffusion_models", "week8_capstone"): "week8_capstone",
    
    # Week 9-12 (optimization_research)
    ("optimization_research", "week9_lora"): "week9_lora",
    ("optimization_research", "week10_moe"): "week10_moe",
    ("optimization_research", "week11_opt"): "week11_opt",
    ("optimization_research", "week12_capstone"): "week12_capstone",
}

def ensure_dirs():
    for d in [COURSE_MAT, INTERACTIVE_PLAT, MODULES_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created: {d}")

def move_content():
    ensure_dirs()
    
    # 1. Handle Week 1 (Special Case: It IS the folder)
    old_week1 = os.path.join(BASE_DIR, "pytorch-tutorial")
    process_folder(old_week1, "week1_pytorch")
    
    # 2. Handle Structured Weeks
    # Iterate over parent categories
    for category in ["llm_foundations", "diffusion_models", "optimization_research"]:
        cat_path = os.path.join(BASE_DIR, category)
        if not os.path.exists(cat_path): continue
        
        for item in os.listdir(cat_path):
            item_path = os.path.join(cat_path, item)
            if os.path.isdir(item_path):
                # Lookup new name
                key = (category, item)
                new_name = WEEK_MAPPINGS.get(key, item) # Default to current name if not mapped
                process_folder(item_path, new_name)

def process_folder(source_path, week_name):
    # Targets
    code_dest = os.path.join(COURSE_MAT, week_name)
    web_dest = os.path.join(MODULES_DIR, week_name)
    
    # Create targets
    if not os.path.exists(code_dest): os.makedirs(code_dest)
    if not os.path.exists(web_dest): os.makedirs(web_dest)
    
    print(f"\nProcessing {week_name}...")
    
    # Move files
    for item in os.listdir(source_path):
        s = os.path.join(source_path, item)
        if not os.path.isfile(s) and not os.path.isdir(s): continue
        
        # Directories (like tests, assets)
        if os.path.isdir(s):
            if item == "tests" or item == "__pycache__":
                shutil.move(s, os.path.join(code_dest, item))
            elif item == "assets":
                shutil.move(s, os.path.join(web_dest, item))
            else:
                # Ambiguous directories -> default to code for now?
                # Actually Week 1 has no other dirs
                pass
            continue
            
        # Files
        ext = os.path.splitext(item)[1]
        
        if ext in CODE_EXT or item == "requirements.txt":
            shutil.move(s, os.path.join(code_dest, item))
            print(f"  [Code] Moved {item}")
        elif ext in WEB_EXT:
            shutil.move(s, os.path.join(web_dest, item))
            print(f"  [Web]  Moved {item}")
        else:
            # Move unknown stuff to code (safe bet) e.g. .txt
            shutil.move(s, os.path.join(code_dest, item))
            print(f"  [Misc] Moved {item}")

if __name__ == "__main__":
    move_content()
    # Create an __init__.py in course_materials to make imports easier
    with open(os.path.join(COURSE_MAT, "__init__.py"), "w") as f:
        f.write("")
    print("\nRestructuring Complete.")
