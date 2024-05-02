document.addEventListener('DOMContentLoaded', function() {
    const btn = document.querySelector('#get-started-btn');
    const uploadInput = document.querySelector('#upload-input');
    const fileFormats = ['csv', 'json'];

    // Event listener for the Get Started button
    btn.addEventListener('click', function() {
        // Trigger file upload input to open
        uploadInput.click();
    });

    // Event listener for the file upload input
    uploadInput.addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            const fileName = file.name.toLowerCase();
            const fileExtension = fileName.substring(fileName.lastIndexOf('.') + 1);

            if (fileFormats.includes(fileExtension)) {
                alert('File uploaded successfully!');
                // Here you can add code to handle the uploaded file, such as parsing it and processing the data
            } else {
                alert('Unsupported file format. Please upload a CSV or JSON file.');
            }
        }
    });

    // Function to animate heading letter by letter
    function animateHeading() {
        const heading = document.querySelector('h1');
        const text = heading.textContent;
        heading.textContent = ''; // Clear the heading text

        // Loop through each character in the text
        for (let i = 0; i < text.length; i++) {
            // Delay each character by 100 milliseconds
            setTimeout(function() {
                heading.textContent += text[i];
            }, i * 100);
        }
    }

    // Call the animateHeading function when the page loads
    animateHeading();
});
