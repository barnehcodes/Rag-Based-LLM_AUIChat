import { useState, useEffect, useRef } from 'react';
import { 
  AppBar, 
  Toolbar, 
  Typography, 
  IconButton, 
  Box, 
  useTheme, 
  Container
} from '@mui/material';
import { 
  GitHub as GitHubIcon, 
  LinkedIn as LinkedInIcon, 
  MenuBook as MenuBookIcon, 
  Brightness4 as Brightness4Icon, 
  Brightness7 as Brightness7Icon 
} from '@mui/icons-material';
import { useColorMode } from '../Theme/ThemeProvider';
import { animate, createScope, stagger } from 'animejs';
import logo from '../../assets/auichat-high-resolution-logo.png';

const Header = () => {
  const theme = useTheme();
  const colorMode = useColorMode();
  const [scrolled, setScrolled] = useState(false);
  const headerRef = useRef(null);
  const scope = useRef(null);

  useEffect(() => {
    const handleScroll = () => {
      const isScrolled = window.scrollY > 10;
      if (isScrolled !== scrolled) {
        setScrolled(isScrolled);
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, [scrolled]);

  useEffect(() => {
    if (!headerRef.current) return;
    
    scope.current = createScope({ root: headerRef }).add(scope => {
      // Animate header icons to fade in with a staggered effect
      animate('.header-icon', {
        scale: [0.9, 1],
        opacity: [0, 1],
        delay: stagger(100),
        easing: 'easeOutElastic(1, .6)',
        duration: 800
      });
    });
    
    return () => scope.current.revert();
  }, []);

  return (
    <AppBar 
      ref={headerRef}
      position="fixed" 
      sx={{
        background: scrolled 
          ? theme.palette.mode === 'dark' 
            ? 'rgba(18, 18, 40, 0.8)' 
            : 'rgba(255, 255, 255, 0.8)' 
          : 'transparent',
        boxShadow: scrolled ? 1 : 0,
        backdropFilter: scrolled ? 'blur(8px)' : 'none',
        transition: 'all 0.3s ease-in-out'
      }}
    >
      <Container maxWidth="xl">
        <Toolbar sx={{ justifyContent: 'space-between' }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <img 
              src={logo} 
              alt="AUIChat Logo" 
              style={{ 
                height: '40px',
                marginRight: '10px',
                transition: 'transform 0.3s ease',
                cursor: 'pointer',
                '&:hover': {
                  transform: 'scale(1.05)'
                }
              }} 
            />
            <Typography 
              variant="h6" 
              sx={{ 
                fontWeight: 'bold',
                display: { xs: 'none', sm: 'block' }
              }}
            >
              AUIChat
            </Typography>
          </Box>
          
          <Box sx={{ display: 'flex' }}>
            <IconButton 
              className="header-icon"
              href="https://github.com" 
              target="_blank"
              color="inherit"
              aria-label="GitHub"
            >
              <GitHubIcon />
            </IconButton>
            <IconButton 
              className="header-icon"
              href="https://linkedin.com" 
              target="_blank"
              color="inherit"
              aria-label="LinkedIn"
            >
              <LinkedInIcon />
            </IconButton>
            <IconButton 
              className="header-icon"
              href="/docs" 
              color="inherit"
              aria-label="Documentation"
            >
              <MenuBookIcon />
            </IconButton>
            <IconButton 
              className="header-icon"
              onClick={colorMode.toggleColorMode} 
              color="inherit"
              aria-label="Toggle light/dark mode"
            >
              {theme.palette.mode === 'dark' ? <Brightness7Icon /> : <Brightness4Icon />}
            </IconButton>
          </Box>
        </Toolbar>
      </Container>
    </AppBar>
  );
};

export default Header;