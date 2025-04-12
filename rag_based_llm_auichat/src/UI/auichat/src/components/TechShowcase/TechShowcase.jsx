import { useEffect, useRef } from 'react';
import { Box, Typography, Container, Paper, useTheme } from '@mui/material';
import { animate, createScope, createSpring, stagger, utils } from 'animejs';

// Tech stack logos/icons
const technologies = [
  { name: 'React', color: '#61DAFB', size: 80 },
  { name: 'MaterialUI', color: '#0081CB', size: 70 },
  { name: 'Three.js', color: '#049EF4', size: 75 },
  { name: 'Anime.js', color: '#FF1461', size: 65 },
  { name: 'Python', color: '#3776AB', size: 80 },
];

const TechShowcase = () => {
  const containerRef = useRef(null);
  const techRef = useRef(null);
  const animScope = useRef(null);
  const theme = useTheme();
  const isDark = theme.palette.mode === 'dark';

  useEffect(() => {
    if (!techRef.current) return;
    
    // Create tech bubbles
    const createBubbles = () => {
      technologies.forEach((tech, index) => {
        const bubble = document.createElement('div');
        bubble.classList.add('tech-bubble');
        bubble.style.width = `${tech.size}px`;
        bubble.style.height = `${tech.size}px`;
        bubble.style.borderRadius = '50%';
        bubble.style.background = `radial-gradient(circle at 30% 30%, ${tech.color}, ${isDark ? '#1a1a2e' : '#f0f0f0'})`;
        bubble.style.boxShadow = `0 0 15px rgba(${isDark ? '255, 255, 255, 0.2' : '0, 0, 0, 0.2'})`;
        bubble.style.display = 'flex';
        bubble.style.alignItems = 'center';
        bubble.style.justifyContent = 'center';
        bubble.style.position = 'absolute';
        bubble.style.cursor = 'pointer';
        bubble.style.fontSize = `${tech.size / 6}px`;
        bubble.style.fontWeight = 'bold';
        bubble.style.color = isDark ? '#ffffff' : '#333333';
        bubble.style.userSelect = 'none';
        bubble.textContent = tech.name;
        
        // Add data attributes for animation
        bubble.dataset.index = index;
        
        techRef.current.appendChild(bubble);
      });

      // Initial positioning of bubbles
      const containerWidth = techRef.current.offsetWidth;
      const containerHeight = 200;
      const bubbles = document.querySelectorAll('.tech-bubble');
      
      bubbles.forEach((bubble, index) => {
        const angle = (index / bubbles.length) * Math.PI * 2;
        const radius = Math.min(containerWidth, containerHeight * 2) / 4;
        const x = containerWidth / 2 + Math.cos(angle) * radius - parseInt(bubble.style.width) / 2;
        const y = containerHeight / 2 + Math.sin(angle) * radius / 1.5 - parseInt(bubble.style.height) / 2;
        
        bubble.style.transform = `translate(${x}px, ${y}px)`;
      });
    };
    
    // Create bubbles and setup animation
    createBubbles();

    if (containerRef.current) {
      animScope.current = createScope({ root: containerRef }).add(scope => {
        // Animate bubbles floating
        animate('.tech-bubble', {
          translateX: () => utils.random(-15, 15),
          translateY: () => utils.random(-15, 15),
          scale: () => [1, 1 + (utils.random(5, 15) / 100)],
          easing: 'easeInOutQuad',
          duration: () => utils.random(2000, 4000),
          delay: stagger(200),
          loop: true,
          direction: 'alternate'
        });
        
        // Add bubble click method
        scope.add('popBubble', (bubble) => {
          // First stop any existing animations on this bubble
          utils.remove(bubble);
          
          // Pop animation on click
          animate({
            targets: bubble,
            scale: [1, 1.3, 1],
            rotate: [0, utils.cleanInlineStylesrandom(-15, 15)],
            duration: 800,
            easing: 'easeOutElastic(1, 0.3)'
          });
        });
      });
      
      // Add click handlers for bubbles
      const bubbles = document.querySelectorAll('.tech-bubble');
      bubbles.forEach(bubble => {
        bubble.addEventListener('click', () => {
          if (animScope.current && animScope.current.methods.popBubble) {
            animScope.current.methods.popBubble(bubble);
          }
        });
      });
    }
    
    // Cleanup function
    return () => {
      if (techRef.current) {
        techRef.current.innerHTML = '';
      }
      if (animScope.current) {
        animScope.current.revert();
      }
    };
  }, [isDark]);

  return (
    <Container ref={containerRef} maxWidth="md" sx={{ mt: 10, mb: 8 }}>
      <Paper 
        elevation={3}
        sx={{
          p: 4,
          borderRadius: 4,
          background: theme.palette.mode === 'dark' 
            ? 'linear-gradient(135deg, rgba(40,10,60,0.7) 0%, rgba(20,20,40,0.8) 100%)' 
            : 'linear-gradient(135deg, rgba(250,250,255,0.9) 0%, rgba(240,240,250,0.8) 100%)',
          backdropFilter: 'blur(10px)',
        }}
      >
        <Typography 
          variant="h4" 
          align="center" 
          sx={{ 
            mb: 4,
            fontWeight: 'bold',
            background: 'linear-gradient(45deg, #9c27b0, #3f51b5)',
            backgroundClip: 'text',
            textFillColor: 'transparent',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}
        >
          Powered by Advanced Technologies
        </Typography>
        
        <Box
          ref={techRef}
          sx={{
            height: '250px',
            position: 'relative',
            mb: 4,
          }}
        />
        
        <Typography 
          variant="body1" 
          align="center"
          sx={{ opacity: 0.8 }}
        >
          AUIChat combines cutting-edge frontend technologies with powerful backend processing to deliver an exceptional user experience. Click on the bubbles to learn more!
        </Typography>
      </Paper>
    </Container>
  );
};

export default TechShowcase;