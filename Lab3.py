import streamlit as st
from openai import OpenAI


#Title
st.title("My Lab 3 Question Answering Chatbot")


#Model selection example
openAI_model = st.sidebar.selectbox(
    "Which Model?",
    ("mini", "regular")
)

if openAI_model == "mini":
    model_to_use = "gpt-4o-mini"
else:
    model_to_use = "gpt-4o"


#To create OpenAI client
if "client" not in st.session_state:
    api_key = st.secrets["OPEN_API_KEY"]
    st.session_state.client = OpenAI(api_key=api_key)


#System prompt
system_message = {
    "role": "system",
    "content": (
        "You are a helpful chatbot. "
        "Answer the user's question in simple language that a 10-year-old can understand. "
        "After answering the question, ask: 'Do you want more info?' "
        "If the user says yes, provide more information and ask again: "
        "'Do you want more info?' "
        "If the user says no, ask: 'What can I help you with?'"
    )
}


#chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "How can I help you?"
        }
    ]
#Show recent chat messages
for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])


#Get user input
if prompt := st.chat_input("What is up?"):

    # Add user message to chat history
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )
    #Show user message
    with st.chat_message("user"):
        st.markdown(prompt)
  #start with an empty buffer
    recent_messages = []

    #User messages in the buffer
    user_message_count = 0


    #start at the newest message and work backwards
    for message in reversed(st.session_state.messages):

        # Only keep user and assistant messages
        if (
            message["role"] == "user"
            or message["role"] == "assistant"
        ):

            #Put the message at the beginning to stay in the right order
            recent_messages.insert(0, message)


            #count user messages
            if message["role"] == "user":
                user_message_count += 1
            #Stop after finding the last two user message
            if user_message_count == 2:
                break


    #To keep the system prompt
    messages_to_send = [system_message] + recent_messages
    #OpenAI client
    client = st.session_state.client


    #To send the recent conversation to openai
    stream = client.chat.completions.create(
        model=model_to_use,
        messages=messages_to_send,
        stream=True
    )


    #Show the assistant response
    with st.chat_message("assistant"):
        response = st.write_stream(stream)


    #Add assistant response to chat history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )