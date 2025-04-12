import { useState, useRef, useEffect } from 'react';
import { 
  Box, 
  Card, 
  TextField, 
  Button, 
  Typography, 
  Paper, 
  Avatar, 
  Chip,
  Stack,
  useTheme,
  Container,
  CircularProgress
} from '@mui/material';
import { Send as SendIcon, SmartToy as BotIcon, Person as PersonIcon } from '@mui/icons-material';
import { animate, createScope, stagger } from 'animejs';

// Updated API URL to use the CORS proxy
const API_URL = 'http://localhost:5001/api';

const ChatMessage = ({ message, isUser }) => {
  const theme = useTheme();
  const messageRef = useRef(null);
  const animScope = useRef(null);

  useEffect(() => {
    if (!messageRef.current) return;
    
    animScope.current = createScope({ root: messageRef }).add(scope => {
      animate(messageRef.current, {
        opacity: [0, 1],
        translateY: [20, 0],
        duration: 400,
        easing: 'easeOutQuad'
      });
    });
    
    return () => {
      if (animScope.current) animScope.current.revert();
    };
  }, []);

  return (
    <Box
      ref={messageRef}
      sx={{
        display: 'flex',
        flexDirection: isUser ? 'row-reverse' : 'row',
        mb: 2,
        opacity: 0,
      }}
    >
      <Avatar
        sx={{
          bgcolor: isUser ? 'primary.main' : 'secondary.main',
          mr: isUser ? 0 : 1,
          ml: isUser ? 1 : 0,
        }}
      >
        {isUser ? <PersonIcon /> : <BotIcon />}
      </Avatar>
      <Paper
        elevation={1}
        sx={{
          p: 2,
          maxWidth: '75%',
          borderRadius: 2,
          backgroundColor: isUser ? 'primary.light' : theme.palette.mode === 'dark' ? 'grey.800' : 'grey.100',
          color: isUser ? 'primary.contrastText' : 'text.primary',
          position: 'relative',
          '&:after': isUser ? {
            content: '""',
            position: 'absolute',
            right: '-10px',
            top: '50%',
            transform: 'translateY(-50%)',
            borderTop: '10px solid transparent',
            borderBottom: '10px solid transparent',
            borderLeft: `10px solid ${theme.palette.primary.light}`,
          } : {
            content: '""',
            position: 'absolute',
            left: '-10px',
            top: '50%',
            transform: 'translateY(-50%)',
            borderTop: '10px solid transparent',
            borderBottom: '10px solid transparent',
            borderRight: `10px solid ${theme.palette.mode === 'dark' ? theme.palette.grey[800] : theme.palette.grey[100]}`,
          },
        }}
      >
        <Typography variant="body1">{message}</Typography>
      </Paper>
    </Box>
  );
};

const ChatMetrics = ({ inferenceTime }) => {
  return (
    <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
      <Chip 
        label={`Inference Time: ${inferenceTime.toFixed(2)}ms`} 
        color="primary" 
        variant="outlined" 
        sx={{ mx: 1 }}
      />
    </Box>
  );
};

const ChatInterface = () => {
  const [messages, setMessages] = useState([
    { text: "Hello! I'm AUIChat. How can I assist you with Al Akhawayn University information today?", isUser: false },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [inferenceTime, setInferenceTime] = useState(0);
  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);
  const animScope = useRef(null);
  const theme = useTheme();
  
  // Scroll to bottom of messages
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);
  
  // Animation for the chat container on mount
  useEffect(() => {
    if (!chatContainerRef.current) return;
    
    animScope.current = createScope({ root: chatContainerRef }).add(scope => {
      animate('.chat-container', {
        translateY: [50, 0],
        opacity: [0, 1],
        duration: 800,
        easing: 'easeOutQuad'
      });
    });
    
    return () => {
      if (animScope.current) animScope.current.revert();
    };
  }, []);

  // Suggested queries
  const suggestedQueries = [
    "What are the requirements for the PiP program?",
    "What counseling services are available at AUI?",
    "How do I apply for undergraduate admission?",
    "What is the transfer student process?",
    "Tell me about Al Akhawayn University."
  ];
  
  const handleSendMessage = async () => {
    if (inputValue.trim() === '') return;
    
    // Add user message
    setMessages(prev => [...prev, { text: inputValue, isUser: true }]);
    const userQuery = inputValue;
    setInputValue('');
    setIsLoading(true);
    
    try {
      // Call the RAG API
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userQuery }),
      });
      
      const data = await response.json();
      
      // Update metrics
      if (data.metrics && data.metrics.inferenceTime) {
        setInferenceTime(data.metrics.inferenceTime);
      }
      
      // Add bot response
      setMessages(prev => [...prev, { text: data.response, isUser: false }]);
    } catch (error) {
      console.error('Error calling RAG API:', error);
      setMessages(prev => [...prev, { 
        text: "I'm sorry, I encountered an error processing your request. Please try again later.", 
        isUser: false 
      }]);
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleSuggestedQuery = (query) => {
    setInputValue(query);
    // Focus on the input field
    document.querySelector('textarea').focus();
  };
  
  return (
    <Container ref={chatContainerRef} maxWidth="md" sx={{ mb: 8, position: 'relative' }}>
      <Card 
        className="chat-container"
        elevation={4} 
        sx={{
          borderRadius: 3,
          overflow: 'hidden',
          backgroundColor: theme.palette.mode === 'dark' ? 'rgba(30, 30, 50, 0.7)' : 'rgba(255, 255, 255, 0.85)',
          backdropFilter: 'blur(8px)',
          transition: 'all 0.3s ease',
          position: 'relative',
          zIndex: 10, // Higher z-index to ensure chat appears above all other elements
          '&:hover': {
            boxShadow: theme.shadows[8],
          },
        }}
      >
        <Box 
          sx={{ 
            p: 2, 
            bgcolor: 'primary.main', 
            color: 'primary.contrastText' 
          }}
        >
          <Typography variant="h6">AUIChat Assistant</Typography>
        </Box>
        
        <ChatMetrics inferenceTime={inferenceTime} />

        {/* Suggested queries */}
        <Box sx={{ px: 2, pb: 2, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
          {suggestedQueries.map((query, index) => (
            <Chip
              key={index}
              label={query}
              variant="outlined"
              onClick={() => handleSuggestedQuery(query)}
              sx={{ 
                cursor: 'pointer',
                '&:hover': { bgcolor: 'primary.light', color: 'primary.contrastText' }
              }}
            />
          ))}
        </Box>
        
        <Box 
          sx={{ 
            height: '350px', 
            overflowY: 'auto', 
            p: 2,
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          {messages.map((message, index) => (
            <ChatMessage 
              key={index} 
              message={message.text} 
              isUser={message.isUser} 
            />
          ))}
          <div ref={messagesEndRef} />
        </Box>
        
        <Box 
          sx={{ 
            p: 2, 
            borderTop: 1, 
            borderColor: 'divider',
            display: 'flex',
            alignItems: 'flex-end', 
          }}
        >
          <TextField
            fullWidth
            multiline
            maxRows={4}
            placeholder="Type your message here..."
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            variant="outlined"
            sx={{ mr: 1 }}
            disabled={isLoading}
          />
          <Button 
            variant="contained" 
            color="primary" 
            endIcon={isLoading ? <CircularProgress size={20} color="inherit" /> : <SendIcon />}
            onClick={handleSendMessage}
            disabled={inputValue.trim() === '' || isLoading}
            sx={{ 
              height: 56,
              borderRadius: 2,
              px: 3, 
              transition: 'all 0.2s ease',
              '&:hover': {
                transform: 'translateY(-2px)',
              }
            }}
          >
            {isLoading ? 'Thinking...' : 'Send'}
          </Button>
        </Box>
      </Card>
    </Container>
  );
};

export default ChatInterface;