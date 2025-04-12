import { useRef } from 'react';
import './App.css';
import { animate, createScope } from 'animejs';
import ThemeProvider from './components/Theme/ThemeProvider';
import Header from './components/Header/Header';
import DynamicBackground from './components/DynamicBackground/DynamicBackground';
import ThreeModel from './components/ThreeModel/ThreeModel';
import ChatInterface from './components/ChatInterface/ChatInterface';
import TechShowcase from './components/TechShowcase/TechShowcase';
import { Box, Container, Typography } from '@mui/material';

function App() {
  return (
    <ThemeProvider>
      {/* Position the background outside the main content but inside ThemeProvider */}
      <DynamicBackground />
      
      <Box sx={{ 
        minHeight: '100vh',
        position: 'relative',
        zIndex: 1 // Ensure content stays above background
      }}>
        <Header />
        {/* Hero section with title and subtitle */}
        <Box 
          sx={{
            pt: { xs: 10, sm: 12 },
            pb: 0,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            textAlign: 'center'
          }}
        >
          <Container maxWidth="lg">
            <Typography 
              variant="h2" 
              sx={{ 
                textAlign: 'center', 
                fontWeight: 'bold',
                mb: 2,
                background: 'linear-gradient(45deg, #9c27b0, #3f51b5)',
                backgroundClip: 'text',
                textFillColor: 'transparent',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              Welcome to AUIChat
            </Typography>
            <Typography 
              variant="h5" 
              sx={{ 
                textAlign: 'center', 
                mb: 4,
                opacity: 0.8,
              }}
            >
              An intelligent chatbot powered by advanced AI technology
            </Typography>
           
            <ThreeModel />
          </Container>
        </Box>
        
        {/* Chat Interface Section - Below the 3D model */}
        <Box 
          sx={{ 
            py: 2,
            position: 'relative',
          }}
        >
          <ChatInterface />
        </Box>
        
        {/* Technology Showcase Section */}
        <Box 
          sx={{ 
            py: 6,
            position: 'relative',
          }}
        >
          <TechShowcase />
        </Box>
        
        {/* Footer */}
        <Box 
          sx={{ 
            py: 3,
            textAlign: 'center',
            opacity: 0.7,
          }}
        >
          <Typography variant="body2">
            © {new Date().getFullYear()} AUIChat. All rights reserved.
          </Typography>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
