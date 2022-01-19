import streamlit as st

st.title("Classification Web App")
st.header("Header")
st.subheader("Subheader")
st.markdown("Markdown")
st.write("Write")
with open("User Manual.pdf", "rb") as file:
    st.download_button("Download here", file, "User Manual.pdf")