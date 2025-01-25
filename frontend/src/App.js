import React, { useState } from "react";
import axios from "axios";
import "./App.css";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

if (!BACKEND_URL) {
  console.error("Backend URL is not defined. Check your environment variables.");
}

function App() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false); // Loading state

  const handleSubmit = async () => {
    if (!query.trim()) {
      alert("Please enter a query.");
      return;
    }

    setIsLoading(true); // Start loading
    setError(null); // Clear previous errors

    try {
      const res = await axios.post(
        `${BACKEND_URL}/query`,
        {
          prompt: query,
          // top_k: 5  // Optional: Make this configurable in the UI if needed
        },
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      setResponse(res.data.response || "No response received.");
      setQuery(""); // Clear the input field
    } catch (err) {
      console.error("Error fetching data:", err);

      // More user-friendly error messages:
      if (err.response) {
        // Server responded with a status code outside 2xx range
        setError(`Server error: ${err.response.status} - ${err.response.data}`);
      } else if (err.request) {
        // Request was made but no response received
        setError("Network error: Could not connect to the server.");
      } else {
        // Something else went wrong
        setError("An unexpected error occurred.");
      }
    } finally {
      setIsLoading(false); // Stop loading
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Memory System Interface</h1>
        <textarea
          placeholder="Type your query here..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={isLoading} // Disable input while loading
        />
        <button onClick={handleSubmit} disabled={isLoading}>
          {isLoading ? "Submitting..." : "Submit Query"}
        </button>

        <div>
          <h2>Response:</h2>
          {isLoading ? (
            <p>Loading...</p> // Show loading message
          ) : error ? (
            <p className="error">{error}</p>
          ) : (
            <p>{response || "No response yet."}</p>
          )}
        </div>
      </header>
    </div>
  );
}

export default App;