import { useEffect, useRef, useState } from 'react';
import { Box, useTheme } from '@mui/material';

const DynamicBackground = () => {
  const containerRef = useRef(null);
  const theme = useTheme();
  const isDark = theme.palette.mode === 'dark';
  const [vantaEffect, setVantaEffect] = useState(null);

  useEffect(() => {
    // Load Three.js and Vanta.js scripts dynamically
    const loadScripts = async () => {
      // Check if Three.js is already loaded
      if (!window.THREE) {
        const threeScript = document.createElement('script');
        threeScript.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js';
        threeScript.async = true;
        document.head.appendChild(threeScript);
        
        await new Promise((resolve) => {
          threeScript.onload = resolve;
        });
      }
      
      // Check if VANTA is already loaded
      if (!window.VANTA) {
        const vantaScript = document.createElement('script');
        vantaScript.src = 'https://cdn.jsdelivr.net/npm/vanta@latest/dist/vanta.globe.min.js';
        vantaScript.async = true;
        document.head.appendChild(vantaScript);
        
        await new Promise((resolve) => {
          vantaScript.onload = resolve;
        });
      }
      
      // Initialize Vanta effect after scripts are loaded
      initVantaEffect();
    };
    
    const initVantaEffect = () => {
      if (window.VANTA && containerRef.current && !vantaEffect) {
        // Destroy any existing effect first to be safe
        if (vantaEffect) {
          vantaEffect.destroy();
        }
        
        // Create new effect
        const effect = window.VANTA.GLOBE({
          el: containerRef.current,
          mouseControls: true,
          touchControls: true,
          gyroControls: false,
          minHeight: 200.00,
          minWidth: 200.00,
          scale: 1.00,
          scaleMobile: 1.00,
          color: isDark ? 0xffffff : 0x3333ff,
          backgroundColor: isDark ? 0x32194d : 0xf0f8ff,
          size: 1.00,
          speed: 0.6,
          points: 8.00
        });
        
        setVantaEffect(effect);
      }
    };

    loadScripts();
    
    // Cleanup function to destroy Vanta effect
    return () => {
      if (vantaEffect) {
        vantaEffect.destroy();
      }
    };
  }, [isDark]);

  // Update the effect when theme changes
  useEffect(() => {
    if (vantaEffect) {
      vantaEffect.destroy();
      setVantaEffect(null);
    }
  }, [isDark]);

  return (
    <Box
      ref={containerRef}
      sx={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        zIndex: -10, // Lower z-index to make sure it's behind everything
        overflow: 'hidden',
        '&::before': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%', 
          zIndex: -9,
          background: isDark ? 'rgba(20, 10, 30, 0.1)' : 'rgba(240, 248, 255, 0.1)',
        }
      }}
    />
  );
};

export default DynamicBackground;