const eventSource = new EventSource("/notifications");
    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        alert(`Accident Detected: ${data.description} at ${data.location}`);
        window.location.href = "/accident_detection";
    };
