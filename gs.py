import requests

def wikipedia_search(query):
    try:
        url = f"https://en.wikipedia.org/w/api.php"
        params = {
            'action': 'query',
            'format': 'json',
            'titles': query,
            'prop': 'extracts',
            'exintro': True,
            'explaintext': True
        }
        response = requests.get(url, params=params)
        data = response.json()
        pages = data['query']['pages']

        # Extract page content
        for page_id, page in pages.items():
            if 'extract' in page:
                extract = page['extract']
                # Split the extract into sentences
                sentences = extract.split('. ')
                # Return the first three sentences
                if len(sentences) > 3:
                    return '. '.join(sentences[:3]) + '.'
                else:
                    return extract

        return "I couldn't find any information about that topic."
    except Exception as e:
        return f"Sorry, I couldn't retrieve the information. Error: {str(e)}"

# Example usage
print(wikipedia_search("Genes"))
