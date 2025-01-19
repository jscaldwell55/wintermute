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
      // Send query to backend
      const res = await axios.post(`${BACKEND_URL}/query`, { query });
      setResponse(res.data.response || "No response received.");
      setError(null); // Clear any previous errors
    } catch (err) {
      console.error("Error fetching data:", err);
      setResponse("");
      setError("An error occurred while fetching data.");
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
