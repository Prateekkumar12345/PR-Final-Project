const API_BASE_URL = 'http://127.0.0.1:8000';

export const sendChatMessage = async (message, sessionId = null) => {
  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ message, session_id: sessionId }),
  });
  
  if (!response.ok) {
    throw new Error('Failed to send message');
  }
  
  return response.json();
};

export const getRecommendations = async (formData) => {
  const response = await fetch(`${API_BASE_URL}/api/recommendations`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ profile: formData }),
  });
  
  if (!response.ok) {
    throw new Error('Failed to get recommendations');
  }
  
  return response.json();
};

export const getSampleRecommendations = async () => {
  const response = await fetch(`${API_BASE_URL}/api/sample-recommendations`);
  
  if (!response.ok) {
    throw new Error('Failed to get sample recommendations');
  }
  
  return response.json();
};

export const getAllColleges = async () => {
  const response = await fetch(`${API_BASE_URL}/api/colleges`);
  
  if (!response.ok) {
    throw new Error('Failed to get colleges');
  }
  
  return response.json();
};

export const getSampleRequest = async () => {
  // This is a mock function for sample data
  return {
    sample_request: {
      grade_10_percentage: "85",
      grade_12_percentage: "85",
      jee_score: "120",
      budget_max: "500000",
      preferred_location: "Karnataka",
      preferred_stream: "Engineering"
    }
  };
};