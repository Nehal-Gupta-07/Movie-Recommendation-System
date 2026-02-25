import ast


def extract_names(json_str):
    """Extract all 'name' values from a JSON-encoded list of dicts."""
    return [item['name'] for item in ast.literal_eval(json_str)]


def extract_top_names(json_str, limit=5):
    """Extract the first `limit` 'name' values from a JSON-encoded list of dicts."""
    return [item['name'] for item in ast.literal_eval(json_str)[:limit]]


def extract_director(json_str):
    """Extract the director's name from a JSON-encoded crew list."""
    for member in ast.literal_eval(json_str):
        if member['job'] == 'Director':
            return [member['name']]
    return []
