import requests

url = " http://127.0.0.1:8000/parameters"
"""with open(
    "F:/projet_programs/kaggles_books/Robotics and computer_project/dataset/seam3/1492.jpg",
    "rb",
) as f:"""
reponse = requests.get(url=url)
print(reponse.json())
