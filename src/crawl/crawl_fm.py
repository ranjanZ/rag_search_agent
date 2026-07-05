import os
import json
import requests
from bs4 import BeautifulSoup

# 1. Setup and create the output directory
os.makedirs('data/raw', exist_ok=True)
faculty_data = []

url = "https://mbzuai.ac.ae/research-department/machine-learning-department/"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

session = requests.Session()
session.headers.update(headers)

# 2. Crawl the main department page to get all faculty profile links
print("Fetching main department page...")
response = session.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

links = soup.find_all('a')
profile_links = set()
for link in links:
    href = link.get('href')
    # Filter for faculty profile URLs
    if href and '/study/faculty/' in href:
        profile_links.add(href)

profile_links = list(profile_links)
print(f"Found {len(profile_links)} unique faculty profile links.")

# 3. Crawl each faculty member's profile page
for i, prof_url in enumerate(profile_links):
    print(f"Processing {i+1}/{len(profile_links)}: {prof_url}")
    try:
        response = session.get(prof_url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract Name (usually in h1)
        h1 = soup.find('h1')
        name = h1.get_text(strip=True) if h1 else "Unknown"
        
        # Extract Title (usually the next sibling of h1)
        title = ""
        if h1:
            next_elem = h1.find_next_sibling()
            if next_elem:
                title = next_elem.get_text(strip=True)
                
        # Extract Email (mailto: link)
        email = ""
        email_tag = soup.find('a', href=lambda x: x and x.startswith('mailto:'))
        if email_tag:
            email = email_tag.get('href').replace('mailto:', '')
            
        # Extract Tabs content (biography, education, accolades, publications, research)
        tabs_data = {}
        profile_content = soup.find('div', class_='profile--content')
        if profile_content:
            # The website uses Alpine.js tabs with x-show attributes
            tabs = profile_content.find_all('div', attrs={'x-show': True})
            for tab in tabs:
                tab_name = tab.get('x-show').split("'")[1]
                # Get clean text content
                content = tab.get_text(separator=' ', strip=True)
                tabs_data[tab_name] = content
                
        faculty_member = {
            "name": name,
            "title": title,
            "email": email,
            "url": prof_url,
            "tabs": tabs_data
        }
        faculty_data.append(faculty_member)
        
        # Save to individual JSON files
        # Clean the name to make it a valid filename
        safe_name = "".join([c for c in name if c.isalpha() or c.isdigit() or c in (' ', '_', '-')]).rstrip()
        filename = f"data/raw/{safe_name.replace(' ', '_').lower()}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(faculty_member, f, indent=4, ensure_ascii=False)
            
    except Exception as e:
        print(f"Error processing {prof_url}: {e}")

# 4. Save all faculty data to a single consolidated JSON file
with open('data/raw/all_faculty.json', 'w', encoding='utf-8') as f:
    json.dump(faculty_data, f, indent=4, ensure_ascii=False)

print("Crawling completed. Data saved to data/raw/")



