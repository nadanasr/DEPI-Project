document.getElementById("send-button").addEventListener("click", sendMessage);
document.getElementById("user-input").addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
});

async function sendMessage() {
    const userInput = document.getElementById("user-input");
    const messageContainer = document.getElementById("message-container");
    
    const userMessage = userInput.value;
    if (userMessage.trim() === "") return;

    // Display user message
    displayMessage(userMessage, "user");
    
    // Clear input
    userInput.value = "";

    // Send user message to Flask backend and get the bot response
    try {
        const response = await fetch('/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ user: userMessage })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        const botResponse = data.response; // Access the response from the server

        // Display bot response
        displayMessage(botResponse, "bot");
    } catch (error) {
        console.error('Error:', error);
    }
}

function displayMessage(message, sender) {
    const messageContainer = document.getElementById("message-container");
    
    const messageDiv = document.createElement("div");
    messageDiv.className = "message " + sender;
    messageDiv.innerText = message;
    
    messageContainer.appendChild(messageDiv);
    messageContainer.scrollTop = messageContainer.scrollHeight; // Auto scroll
}

