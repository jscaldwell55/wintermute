import React, { useState } from "react";
import axios from "axios";
import "./App.css";

// Backend URL setup
const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

if (!BACKEND_URL) {
  console.error("Backend URL is not defined. Check your environment variables.");
}
  
function App() {
  const [query, setQuery] = useState(""); // User input query
  const [response, setResponse] = useState(""); // Backend response
  const [error, setError] = useState(null); // Error handling

  // Function to handle query submission
  const handleSubmit = async () => {
    if (!query.trim()) {
      alert("Please enter a query.");
      return;
    }
  
    try {
      const res = await axios.post(
        `${BACKEND_URL}/query`, // Construct the API endpoint URL
        { 
          prompt: query, // Data payload for the backend
          top_k: 5       // Optional parameter for backend processing
        },
        {
          headers: {
            "Content-Type": "application/json", // Ensure proper request header
          },
        }
      );
  
      // Set response state
      setResponse(res.data.response || "No response received.");
      setError(null); // Clear any previous errors
    } catch (err) {
      // Handle error and log details for debugging
      console.error("Error fetching data:", err);
      setResponse("An error occurred while fetching data.");
      setError(err.message || "An unknown error occurred.");
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
        />
        <button onClick={handleSubmit}>Submit Query</button>

        <div>
          <h2>Response:</h2>
          {error ? (
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
