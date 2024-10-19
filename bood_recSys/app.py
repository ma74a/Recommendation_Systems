import pickle
import requests
import numpy as np
import pandas as pd
import streamlit as st

st.header('Book Recommendation System')

try:
    # Load CSV file
    df = pd.read_csv("data/books_info.csv")

    # Load model and other files using with open
    with open('files/model.pkl', 'rb') as model:
        model = pickle.load(model)

    with open('files/books_name.pkl', 'rb') as book_names:
        book_names = pickle.load(book_names)

    with open('files/book_pivot.pkl', 'rb') as book_pivot:
        book_pivot = pickle.load(book_pivot)

except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.stop()

except Exception as e:
    st.error(f"An error occurred while loading the files: {e}")
    st.stop()


headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
}

# define get posters functioin:
def get_posters(indices):
    imgs_url_lst = []
    for idx in indices:
        url = df['Image-URL-M'].iloc[idx]
        imgs_url_lst.append(url)
    posters_lst = []
    for link in imgs_url_lst:
        try:
            response = requests.get(link, headers=headers)
            posters_lst.append(response.url)
        except requests.RequestException as e:
            st.error(f"Failed to fetch poster image: {e}")
    return posters_lst

# define recommend function
def recommend_books(book_name):
    book_id = np.where(book_pivot.index == book_name)[0][0]
    _, indices = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1, -1), n_neighbors=6)
    books_names_lst = []
    books_posters_lst = get_posters(indices[0])
    for i in range(len(indices)):
        book = book_pivot.index[indices[i]]
        for u in book:
            books_names_lst.append(u)
    return books_names_lst, books_posters_lst

selected_book = st.selectbox(
    "Type or select a book to recommend: ",
    book_names
)

if st.button("show recommendation"):
    recommended_books, poster_url = recommend_books(selected_book)
    if recommended_books and poster_url:
        try:
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.text(recommended_books[1])
                st.image(poster_url[1])
            with col2:
                st.text(recommended_books[2])
                st.image(poster_url[2])
            with col3:
                st.text(recommended_books[3])
                st.image(poster_url[3])
            with col4:
                st.text(recommended_books[4])
                st.image(poster_url[4])
            with col5:
                st.text(recommended_books[5])
                st.image(poster_url[5])
        except IndexError as e:
            st.error(f"An error occurred while displaying recommendations: {e}")