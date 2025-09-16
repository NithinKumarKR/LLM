import requests

import streamlit as st

def get_openai_response(input_text):
    response  =  requests.post("")