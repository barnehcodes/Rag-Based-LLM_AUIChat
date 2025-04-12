import { createContext, useContext, useState, useMemo } from 'react';
import { ThemeProvider as MUIThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline } from '@mui/material';

const ColorModeContext = createContext({ toggleColorMode: () => {} });

export const useColorMode = () => useContext(ColorModeContext);

export default function ThemeProvider({ children }) {
  const [mode, setMode] = useState('dark'); // Default to dark mode

  const colorMode = useMemo(
    () => ({
      toggleColorMode: () => {
        setMode((prevMode) => (prevMode === 'light' ? 'dark' : 'light'));
      },
      mode,
    }),
    [mode],
  );

  const theme = useMemo(
    () =>
      createTheme({
        palette: {
          mode,
          primary: {
            main: '#6a1b9a', // Deep purple
          },
          secondary: {
            main: '#3949ab', // Indigo blue
          },
          background: {
            default: 'transparent', // Make default background transparent
            paper: mode === 'dark' ? 'rgba(18, 18, 40, 0.85)' : 'rgba(255, 255, 255, 0.85)', // Semi-transparent backgrounds
          },
        },
        typography: {
          fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
        },
        components: {
          MuiCssBaseline: {
            styleOverrides: {
              body: {
                overflowX: 'hidden',
                scrollBehavior: 'smooth',
                backgroundColor: mode === 'dark' ? 'rgba(10, 10, 26, 0.5)' : 'rgba(245, 245, 245, 0.5)', // Semi-transparent body background
              },
            },
          },
          // Add semi-transparent styling for Paper components
          MuiPaper: {
            styleOverrides: {
              root: {
                backgroundColor: mode === 'dark' ? 'rgba(18, 18, 40, 0.85)' : 'rgba(255, 255, 255, 0.85)',
                backdropFilter: 'blur(5px)', // Add slight blur effect for better readability
              },
            },
          },
        },
      }),
    [mode],
  );

  return (
    <ColorModeContext.Provider value={colorMode}>
      <MUIThemeProvider theme={theme}>
        <CssBaseline />
        {children}
      </MUIThemeProvider>
    </ColorModeContext.Provider>
  );
}