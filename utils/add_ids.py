import json

def add_ids_to_faculty_json(input_path, output_path=None, start_id=0):
    """
    Reads a JSON file containing a list of objects,
    adds an integer 'id' field to each entry (starting from start_id),
    and writes the updated list to a new file (or overwrites if no output_path).
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    # Add id based on index + start_id
    for idx, item in enumerate(data_list):
        item['id'] = start_id + idx

    # Determine output file
    if output_path is None:
        output_path = input_path  # overwrite

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)

    print(f"✅ Added 'id' to {len(data_list)} entries (starting from {start_id}). Saved to {output_path}")

# Example usage:
add_ids_to_faculty_json('data/raw/all_faculty.json', 'data/raw/all_faculty_with_ids.json', start_id=0)
add_ids_to_faculty_json('data/raw/research_projects.json', 'data/raw/research_projects_with_ids.json', start_id=100)