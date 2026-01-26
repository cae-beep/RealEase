// ai_chatbot.js - FIXED VERSION with proper abort handling
import { getAuth } from "https://www.gstatic.com/firebasejs/12.4.0/firebase-auth.js";

// Your Python backend URL - LOCAL DEVELOPMENT
const PYTHON_CHAT_API = "http://localhost:5000/api/chat";

// Global abort controller and timeout
let currentController = null;
let currentTimeout = null;

// Initialize chatbot in your dashboard
export function initChatbot() {
    console.log("🤖 AI Chatbot Initializing...");
    
    const chatInput = document.getElementById('chatInput');
    const sendChatBtn = document.getElementById('sendChatBtn');
    const voiceInputBtn = document.getElementById('voiceInputBtn');
    const chatMessages = document.getElementById('chatMessages');
    
    if (!chatInput || !sendChatBtn || !chatMessages) {
        console.error("❌ Chatbot elements not found!");
        return;
    }
    
    // Show welcome message on first load
    showWelcomeMessage();
    
    // Send message on button click
    sendChatBtn.addEventListener('click', async () => {
        const message = chatInput.value.trim();
        if (message) {
            await processChatMessage(message);
            chatInput.value = '';
        }
    });
    
    // Send message on Enter key
    chatInput.addEventListener('keypress', async (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            const message = chatInput.value.trim();
            if (message) {
                await processChatMessage(message);
                chatInput.value = '';
            }
        }
    });
    
    // Voice input button (optional)
    if (voiceInputBtn) {
        voiceInputBtn.addEventListener('click', () => {
            alert("Voice input would require additional setup with Web Speech API");
        });
    }
    
    console.log("✅ AI Chatbot Initialized!");
}

// Clean up any ongoing requests
function cleanupCurrentRequest() {
    if (currentTimeout) {
        clearTimeout(currentTimeout);
        currentTimeout = null;
    }
    
    if (currentController) {
        currentController.abort();
        currentController = null;
    }
}

// Main function to process chat messages - FIXED VERSION
export async function processChatMessage(userMessage) {
    try {
        const auth = getAuth();
        const currentUser = auth.currentUser;
        
        // Clean up any existing request first
        cleanupCurrentRequest();
        
        // Add user message to chat
        addMessageToChat(userMessage, 'user');
        
        // Show typing indicator
        const typingMessage = addTypingIndicator();
        
        // Prepare request to Python backend
        const requestData = {
            query: userMessage,
            user_id: currentUser ? currentUser.uid : 'anonymous'
        };
        
        console.log("📤 Sending to Python backend:", requestData);
        
        let data;
        try {
            // Create new abort controller
            currentController = new AbortController();
            
            // Set a timeout to abort the request
            currentTimeout = setTimeout(() => {
                if (currentController) {
                    console.log('⏰ Request timeout - aborting');
                    currentController.abort();
                }
            }, 60000); // 60 second timeout
            
            const response = await fetch(PYTHON_CHAT_API, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(requestData),
                signal: currentController.signal,
                mode: 'cors',
                credentials: 'omit'
            });
            
            // Clear timeout since we got a response
            clearTimeout(currentTimeout);
            currentTimeout = null;
            
            if (!response.ok) {
                console.error('HTTP Error:', response.status, response.statusText);
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            // First check if response is JSON
            const contentType = response.headers.get('content-type');
            if (!contentType || !contentType.includes('application/json')) {
                console.warn('Response is not JSON:', contentType);
                const text = await response.text();
                console.log('Raw text response:', text.substring(0, 200));
                throw new Error('Invalid response format from server');
            }
            
            data = await response.json();
            console.log("✅ Successfully received data:", data);
            
        } catch (fetchError) {
            console.error('Fetch error details:', fetchError);
            
            // Clean up timeout and controller
            if (currentTimeout) {
                clearTimeout(currentTimeout);
                currentTimeout = null;
            }
            currentController = null;
            
            // Check specific error types
            if (fetchError.name === 'AbortError') {
                console.warn('Request was aborted (timeout or user action)');
                data = {
                    success: false,
                    response: "The request took too long to process. Please try again with a simpler query or check if the Python backend is running.",
                    properties: [],
                    intent: 'timeout',
                    properties_found: 0
                };
            } else if (fetchError.message.includes('Failed to fetch') || 
                      fetchError.message.includes('NetworkError') ||
                      fetchError.message.includes('ERR_CONNECTION_REFUSED')) {
                console.warn('Network error - Python backend may not be running');
                data = {
                    success: false,
                    response: `I received your query: "${userMessage}". The AI backend is currently unavailable. Please make sure your Python server is running on localhost:5000.`,
                    properties: [],
                    intent: 'backend_error',
                    properties_found: 0
                };
            } else {
                // Use fallback response for other errors
                data = {
                    success: true,
                    response: `I received your query: "${userMessage}". Here are some properties you might like in Batangas.`,
                    properties: getFallbackProperties(userMessage),
                    intent: 'fallback',
                    properties_found: getFallbackProperties(userMessage).length
                };
            }
        }
        
        // Remove typing indicator
        if (typingMessage && typingMessage.remove) {
            typingMessage.remove();
        }
        
        // Reset controller and timeout
        currentController = null;
        currentTimeout = null;
        
        // Remove demo prompts when user sends a message
        const demoPrompts = document.querySelector('.demo-prompts');
        if (demoPrompts) {
            demoPrompts.remove();
        }
        
        // Display response
        if (data && data.response) {
            addMessageToChat(data.response, 'bot');
        } else {
            addMessageToChat("I'm here to help! Try asking about properties in Batangas.", 'bot');
        }
        
        // If properties were found, display them
        if (data.properties && data.properties.length > 0) {
            displayPropertiesInChat(data.properties);
        }
        
        // Show demo prompts again after response
        setTimeout(addDemoPrompts, 500);
        
        // Try to log (non-critical)
        try {
            if (data.success !== false && data.intent !== 'timeout' && data.intent !== 'backend_error') {
                await logChatInteraction(userMessage, data, currentUser);
            }
        } catch (logError) {
            console.log('Non-critical log error:', logError.message);
        }
        
    } catch (error) {
        console.error('Unexpected error in processChatMessage:', error);
        
        // Clean up if still exists
        cleanupCurrentRequest();
        
        // Remove typing indicator
        const typingIndicator = document.querySelector('.typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
        
        // Remove demo prompts on error
        document.querySelector('.demo-prompts')?.remove();
        
        // Show user-friendly error
        addMessageToChat(
            "I'm having some technical difficulties. Please try asking again or use the search filters above.", 
            'bot'
        );
        
        // Show demo prompts again after error
        setTimeout(addDemoPrompts, 500);
    }
}

// Helper function for fallback properties
function getFallbackProperties(query) {
    const fallbackProperties = [
        {
            id: 'fallback-1',
            title: 'Spacious Family Home in Batangas City',
            address: 'Batangas City Proper',
            bedrooms: 3,
            floorArea: 120,
            monthlyRent: 15000,
            photos: ['https://via.placeholder.com/200x150/667eea/ffffff?text=Family+Home']
        },
        {
            id: 'fallback-2',
            title: 'Modern Apartment with Parking',
            address: 'Lipa City',
            bedrooms: 2,
            floorArea: 75,
            monthlyRent: 12000,
            photos: ['https://via.placeholder.com/200x150/4CAF50/ffffff?text=Modern+Apt']
        },
        {
            id: 'fallback-3',
            title: 'Affordable Studio Unit',
            address: 'Tanauan City',
            bedrooms: 'Studio',
            floorArea: 35,
            monthlyRent: 8000,
            photos: ['https://via.placeholder.com/200x150/FF9800/ffffff?text=Studio']
        }
    ];
    
    // Filter based on query keywords
    const lowerQuery = query.toLowerCase();
    if (lowerQuery.includes('family')) {
        return [fallbackProperties[0]];
    } else if (lowerQuery.includes('parking') || lowerQuery.includes('apartment')) {
        return [fallbackProperties[1]];
    } else if (lowerQuery.includes('cheap') || lowerQuery.includes('budget') || lowerQuery.includes('studio')) {
        return [fallbackProperties[2]];
    }
    
    return fallbackProperties;
}

// Add messages to chat UI
function addMessageToChat(message, sender) {
    const chatMessages = document.getElementById('chatMessages');
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    const avatar = sender === 'user' ? '👤' : '🤖';
    
    messageDiv.innerHTML = `
        <div class="avatar">${avatar}</div>
        <div class="content">
            <p>${message}</p>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Add typing indicator
function addTypingIndicator() {
    const chatMessages = document.getElementById('chatMessages');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot typing-indicator';
    typingDiv.innerHTML = `
        <div class="avatar">🤖</div>
        <div class="content">
            <div class="typing">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return typingDiv;
}

// Display properties in chat
function displayPropertiesInChat(properties) {
    const chatMessages = document.getElementById('chatMessages');
    
    const propertiesDiv = document.createElement('div');
    propertiesDiv.className = 'chat-properties-container';
    
    let html = '<div class="properties-grid">';
    
    // Show max 3 properties in chat
    properties.slice(0, 3).forEach(prop => {
        const price = getDisplayPrice(prop);
        const bedrooms = prop.bedrooms || 'N/A';
        const area = prop.floorArea || prop.totalArea || 'N/A';
        const photo = prop.photos?.[0] || prop.imageUrls?.[0] || 'https://via.placeholder.com/200x150';
        
        html += `
            <div class="property-card-chat">
                <div class="property-image">
                    <img src="${photo}" alt="${prop.title}" onerror="this.src='https://via.placeholder.com/200x150'">
                </div>
                <div class="property-info">
                    <h4>${prop.title || 'Untitled Property'}</h4>
                    <p class="location">📍 ${prop.address || prop.city || 'Location not specified'}</p>
                    <div class="details">
                        <span>🛏️ ${bedrooms} ${bedrooms === 'Studio' ? '' : 'beds'}</span>
                        ${area && area !== 'N/A' ? `<span>📐 ${area} sqm</span>` : ''}
                    </div>
                    <p class="price">${price}</p>
                    <a href="property_details.html?id=${prop.id}" target="_blank" class="view-btn">View Details</a>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    
    if (properties.length > 3) {
        html += `<p style="text-align: center; margin-top: 10px;">
                    <a href="search_results.html" style="color: var(--primary); text-decoration: underline;">
                        View all ${properties.length} properties →
                    </a>
                 </p>`;
    }
    
    propertiesDiv.innerHTML = html;
    
    chatMessages.appendChild(propertiesDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Helper function to format price
function getDisplayPrice(property) {
    if (property.monthlyRent) {
        return `₱${property.monthlyRent.toLocaleString()}/month`;
    } else if (property.annualRent) {
        return `₱${property.annualRent.toLocaleString()}/year`;
    } else if (property.salePrice) {
        return `₱${property.salePrice.toLocaleString()}`;
    } else if (property.pricing) {
        return `₱${property.pricing.toLocaleString()}`;
    }
    return 'Price on inquiry';
}

// Log chat interactions (optional)
async function logChatInteraction(query, response, user) {
    try {
        if (!user) return;
        
        // Import Firestore inside function to avoid initialization issues
        const { getFirestore, collection, addDoc } = await import("https://www.gstatic.com/firebasejs/12.4.0/firebase-firestore.js");
        const { getApp } = await import("https://www.gstatic.com/firebasejs/12.4.0/firebase-app.js");
        
        // Get initialized app and Firestore
        const app = getApp();
        const db = getFirestore(app);
        
        await addDoc(collection(db, 'chatbot_logs'), {
            userId: user.uid,
            query: query,
            intent: response.intent || 'unknown',
            entities: response.entities || {},
            response: response.response?.substring(0, 200) || '',
            propertiesFound: response.properties_found || 0,
            timestamp: new Date(),
            modelUsed: response.model_used || 'unknown',
            confidence: response.confidence || 0
        });
        
        console.log('✅ Chat interaction logged');
    } catch (error) {
        console.log('Could not log chat interaction (non-critical):', error.message);
        // This is non-critical, so don't throw error
    }
}

// Rest of the functions remain the same...
// [Keep addDemoPrompts, showWelcomeMessage, and CSS styles as they were]
// Add demo prompts to chat interface covering all 10 questions
function addDemoPrompts() {
    const chatInput = document.getElementById('chatInput');
    const chatMessages = document.getElementById('chatMessages');
    
    if (!chatInput || !chatMessages) return;
    
    // Remove existing demo prompts if any
    const existingPrompts = document.querySelector('.demo-prompts');
    if (existingPrompts) {
        existingPrompts.remove();
    }
    
    // Quick prompts covering ALL 10 questions
    const quickPrompts = [
        // Question 1 (Member 1) - Basic search
        {
            text: "Find apartments in Batangas City",
            question: 1,
            member: "member1",
            emoji: "🏢"
        },
        // Question 2 (Member 2) - Detailed criteria
        {
            text: "Show me houses under 3M with 3 bedrooms",
            question: 2,
            member: "member2",
            emoji: "🏠"
        },
        // Question 3 (Member 3) - Family needs
        {
            text: "Find properties for family needs in Lipa City",
            question: 3,
            member: "member3",
            emoji: "👨‍👩‍👧‍👦"
        },
        // Question 4 (Member 2) - Near landmarks
        {
            text: "Properties near hospitals in Tanauan",
            question: 4,
            member: "member2",
            emoji: "🏥"
        },
        // Question 5 (Member 3) - Features with price
        {
            text: "Show me apartments with parking at good price",
            question: 5,
            member: "member3",
            emoji: "🚗"
        },
        // Question 6 (Member 2) - Ready to move
        {
            text: "Find ready to move in properties for students in Batangas City",
            question: 6,
            member: "member2",
            emoji: "🎓"
        },
        // Question 7 (Member 1) - Financing
        {
            text: "Properties that accept Pag-IBIG financing",
            question: 7,
            member: "member1",
            emoji: "💰"
        },
        // Question 8 (Member 3) - Process info
        {
            text: "Steps for buying a condo",
            question: 8,
            member: "member3",
            emoji: "📋"
        },
        // Question 9 (Member 1) - Location info
        {
            text: "Tell me about Nasugbu",
            question: 9,
            member: "member1",
            emoji: "📍"
        },
        // Question 10 (Member 3) - Lifestyle match
        {
            text: "What properties match my budget as a single professional?",
            question: 10,
            member: "member3",
            emoji: "🎯"
        }
    ];
    
    // Shuffle the prompts for variety
    const shuffledPrompts = [...quickPrompts].sort(() => Math.random() - 0.5);
    
    // Create demo prompts section
    const demoSection = document.createElement('div');
    demoSection.className = 'demo-prompts';
    demoSection.style.cssText = `
        margin-top: 15px;
        padding: 15px;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        animation: fadeIn 0.5s ease;
    `;
    
    let buttonsHTML = '';
    // Show 6 random prompts from the shuffled list
    for (let i = 0; i < 6 && i < shuffledPrompts.length; i++) {
        const prompt = shuffledPrompts[i];
        buttonsHTML += `
            <button class="demo-prompt-btn" data-prompt="${prompt.text}">
                ${prompt.emoji} ${prompt.text}
            </button>
        `;
    }
    
    demoSection.innerHTML = `
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; margin-bottom: 12px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="font-size: 14px; color: var(--text-dark); font-weight: 600;">
                    <i class="fas fa-bolt"></i> Quick Prompts
                </div>
                <div style="font-size: 11px; color: #666; background: rgba(255,255,255,0.7); padding: 2px 8px; border-radius: 10px;">
                    All 10 Questions Covered
                </div>
            </div>
            <div style="font-size: 11px; color: #888;">
                Click any prompt to try
            </div>
        </div>
        <div style="display: flex; overflow-x: auto; gap: 10px; padding-bottom: 10px; scrollbar-width: thin;">
            ${buttonsHTML}
        </div>
        <div style="margin-top: 8px; font-size: 11px; color: #888; text-align: center;">
            Prompts change on refresh
        </div>
    `;
    
    chatMessages.parentNode.insertBefore(demoSection, chatMessages.nextSibling);
    
    // Add event listeners to demo prompt buttons
    setTimeout(() => {
        document.querySelectorAll('.demo-prompt-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const prompt = this.getAttribute('data-prompt');
                chatInput.value = prompt;
                chatInput.focus();
                
                // Highlight the button briefly
                this.style.transform = 'scale(0.95)';
                this.style.boxShadow = '0 0 0 2px rgba(102, 126, 234, 0.3)';
                setTimeout(() => {
                    this.style.transform = '';
                    this.style.boxShadow = '';
                }, 300);
            });
        });
    }, 100);
}

// Show welcome message on first load
function showWelcomeMessage() {
    const chatMessages = document.getElementById('chatMessages');
    if (chatMessages && chatMessages.children.length === 0) {
        setTimeout(() => {
            const welcomeMessage = `
                <div class="welcome-message">
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
                        <div style="width: 50px; height: 50px; border-radius: 50%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            display: flex; align-items: center; justify-content: center; font-size: 24px;">
                            🤖
                        </div>
                        <div>
                            <h4 style="margin: 0; color: var(--text-dark);">AI Property Assistant</h4>
                            <p style="margin: 0; font-size: 12px; color: #666;">Specialized in Batangas Properties</p>
                        </div>
                    </div>
                    <p style="color: var(--text-dark); margin-bottom: 15px;">
                        Hello! I'm your AI property assistant for Batangas. I can help you with all 10 types of property questions:
                    </p>
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 20px;">
                        <div style="background: white; padding: 10px; border-radius: 8px; border: 1px solid var(--border);">
                            <div style="font-weight: 600; color: var(--primary);">Q1</div>
                            <div style="font-size: 11px;">Basic property search</div>
                        </div>
                        <div style="background: white; padding: 10px; border-radius: 8px; border: 1px solid var(--border);">
                            <div style="font-weight: 600; color: var(--primary);">Q2</div>
                            <div style="font-size: 11px;">Detailed criteria</div>
                        </div>
                        <div style="background: white; padding: 10px; border-radius: 8px; border: 1px solid var(--border);">
                            <div style="font-weight: 600; color: var(--primary);">Q3-6</div>
                            <div style="font-size: 11px;">Special needs & features</div>
                        </div>
                        <div style="background: white; padding: 10px; border-radius: 8px; border: 1px solid var(--border);">
                            <div style="font-weight: 600; color: var(--primary);">Q7-10</div>
                            <div style="font-size: 11px;">Financing & lifestyle</div>
                        </div>
                    </div>
                    <p style="color: var(--text-dark); margin-bottom: 10px;">
                        <strong>Try asking about:</strong>
                    </p>
                    <ul style="color: var(--text-dark); font-size: 13px; margin: 0 0 15px 15px; padding: 0;">
                        <li>Finding specific properties</li>
                        <li>Financing options & documents</li>
                        <li>Location information</li>
                        <li>Property features & amenities</li>
                    </ul>
                    <p style="color: #666; font-size: 12px; font-style: italic;">
                        <i class="fas fa-lightbulb"></i> Try the quick prompts below to get started!
                    </p>
                </div>
            `;
            
            const welcomeDiv = document.createElement('div');
            welcomeDiv.className = 'message bot';
            welcomeDiv.innerHTML = `
                <div class="avatar">🤖</div>
                <div class="content">${welcomeMessage}</div>
            `;
            chatMessages.appendChild(welcomeDiv);
            
            // Show demo prompts after welcome message
            setTimeout(addDemoPrompts, 500);
        }, 300);
    }
}

// Make functions available globally
window.processChatMessage = processChatMessage;
window.initChatbot = initChatbot;

console.log("🚀 AI Chatbot Script Loaded Successfully!");