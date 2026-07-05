import requests
from bs4 import BeautifulSoup
import csv
import json

def crawl_research_projects(url):
    # Set a standard User-Agent to avoid being blocked by basic bot protections
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    }
    
    print(f"Fetching data from {url} ...")
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Failed to retrieve page. Status code: {response.status_code}")
        return []

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Based on the site's structure, each project is contained in a div with class 'div-block-41'
    project_blocks = soup.find_all('div', class_='div-block-41')
    projects_data = []
    
    for block in project_blocks:
        # Extract text from specific IDs within the block
        title = block.find('div', id='title').get_text(strip=True) if block.find('div', id='title') else "N/A"
        summary = block.find('div', id='summary').get_text(strip=True) if block.find('div', id='summary') else "N/A"
        owner = block.find('div', id='owner').get_text(strip=True) if block.find('div', id='owner') else "N/A"
        category = block.find('div', id='category').get_text(strip=True) if block.find('div', id='category') else "N/A"
        
        # The description contains HTML paragraphs (<p>), so we extract them line-by-line
        desc_div = block.find('div', id='description')
        if desc_div:
            description = '\n'.join([p.get_text(strip=True) for p in desc_div.find_all('p')])
            if not description:
                description = desc_div.get_text(strip=True)
        else:
            description = "N/A"
        
        projects_data.append({
            "Title": title,
            "Summary": summary,
            "Owner": owner,
            "Category": category,
            "Description": description
        })
        
    return projects_data

def save_to_csv(data, filename="data/raw/research_projects.csv"):
    if not data: return
    keys = data[0].keys()
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)
    print(f"✅ Data successfully saved to {filename}")

def save_to_json(data, filename="data/raw/research_projects.json"):
    if not data: return
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"✅ Data successfully saved to {filename}")

if __name__ == "__main__":
    target_url = "https://research.mbzuai.ac.ae/research-projects"
    projects = crawl_research_projects(target_url)
    
    if projects:
        print(f"🎉 Successfully extracted {len(projects)} research projects.")
        save_to_csv(projects)
        save_to_json(projects)
    else:
        print("❌ No projects found.")


