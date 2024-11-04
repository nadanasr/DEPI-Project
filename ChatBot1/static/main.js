// JavaScript to handle form submission via AJAX
document.getElementById('chat-form').addEventListener('submit', async function (e) {
    e.preventDefault();  // Prevent the default form submission behavior
    
    let form = e.target;
    let formData = new FormData(form);
    
    // Send an asynchronous request using Fetch API
    let response = await fetch('/get', {
        method: 'POST',
        body: formData
    });

    if (response.ok) {
        let result = await response.text();  // Get the response text (HTML)
        
        // Parse the HTML and update the response div
        let parser = new DOMParser();
        let htmlDoc = parser.parseFromString(result, 'text/html');
        let newResponse = htmlDoc.querySelector('#result').innerHTML;
        
        // Update the result div with the new response
        document.getElementById('result').innerHTML = newResponse;
        
        // Clear the input field
        document.getElementById('inputName').value = '';
    } else {
        console.error('Error:', response.statusText);
    }
});
