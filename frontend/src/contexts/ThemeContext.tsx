import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { createTheme, ThemeProvider, Theme } from '@mui/material/styles';

// Types
type ThemeMode = 'light' | 'dark' | 'auto';

interface ThemeContextType {
  mode: ThemeMode;
  theme: Theme;
  toggleTheme: () => void;
  setThemeMode: (mode: ThemeMode) => void;
  isDarkMode: boolean;
}

// Create context
const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

// Custom hook to use theme context
export const useTheme = (): ThemeContextType => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeContextProvider');
  }
  return context;
};

// Theme provider component
interface ThemeContextProviderProps {
  children: ReactNode;
}

export const ThemeContextProvider: React.FC<ThemeContextProviderProps> = ({ children }) => {
  const [mode, setMode] = useState<ThemeMode>('light');
  const [systemPrefersDark, setSystemPrefersDark] = useState(false);

  // Check system preference
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    setSystemPrefersDark(mediaQuery.matches);

    const handleChange = (e: MediaQueryListEvent) => {
      setSystemPrefersDark(e.matches);
    };

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  // Load saved theme preference
  useEffect(() => {
    const savedMode = localStorage.getItem('theme_mode') as ThemeMode;
    if (savedMode && ['light', 'dark', 'auto'].includes(savedMode)) {
      setMode(savedMode);
    }
  }, []);

  // Determine if dark mode should be active
  const isDarkMode = mode === 'dark' || (mode === 'auto' && systemPrefersDark);

  // Create theme based on mode
  const theme = createTheme({
    palette: {
      mode: isDarkMode ? 'dark' : 'light',
      primary: {
        main: '#1976d2',
        light: '#42a5f5',
        dark: '#1565c0',
      },
      secondary: {
        main: '#9c27b0',
        light: '#ba68c8',
        dark: '#7b1fa2',
      },
      background: {
        default: isDarkMode ? '#121212' : '#fafafa',
        paper: isDarkMode ? '#1e1e1e' : '#ffffff',
      },
      text: {
        primary: isDarkMode ? '#ffffff' : '#212121',
        secondary: isDarkMode ? '#b3b3b3' : '#757575',
      },
    },
    typography: {
      fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
      h1: {
        fontWeight: 600,
        fontSize: '2.5rem',
      },
      h2: {
        fontWeight: 600,
        fontSize: '2rem',
      },
      h3: {
        fontWeight: 600,
        fontSize: '1.75rem',
      },
      h4: {
        fontWeight: 600,
        fontSize: '1.5rem',
      },
      h5: {
        fontWeight: 600,
        fontSize: '1.25rem',
      },
      h6: {
        fontWeight: 600,
        fontSize: '1rem',
      },
      body1: {
        fontSize: '1rem',
        lineHeight: 1.6,
      },
      body2: {
        fontSize: '0.875rem',
        lineHeight: 1.6,
      },
      button: {
        textTransform: 'none',
        fontWeight: 500,
      },
    },
    shape: {
      borderRadius: 8,
    },
    components: {
      MuiButton: {
        styleOverrides: {
          root: {
            borderRadius: 8,
            padding: '8px 16px',
          },
          contained: {
            boxShadow: isDarkMode 
              ? '0 2px 4px rgba(0,0,0,0.3)' 
              : '0 2px 4px rgba(0,0,0,0.1)',
            '&:hover': {
              boxShadow: isDarkMode 
                ? '0 4px 8px rgba(0,0,0,0.4)' 
                : '0 4px 8px rgba(0,0,0,0.15)',
            },
          },
        },
      },
      MuiCard: {
        styleOverrides: {
          root: {
            boxShadow: isDarkMode 
              ? '0 2px 8px rgba(0,0,0,0.3)' 
              : '0 2px 8px rgba(0,0,0,0.1)',
            borderRadius: 12,
            backgroundColor: isDarkMode ? '#1e1e1e' : '#ffffff',
          },
        },
      },
      MuiPaper: {
        styleOverrides: {
          root: {
            borderRadius: 8,
            backgroundColor: isDarkMode ? '#1e1e1e' : '#ffffff',
          },
        },
      },
      MuiTextField: {
        styleOverrides: {
          root: {
            '& .MuiOutlinedInput-root': {
              borderRadius: 8,
              backgroundColor: isDarkMode ? '#2a2a2a' : '#ffffff',
              '& fieldset': {
                borderColor: isDarkMode ? '#404040' : '#e0e0e0',
              },
              '&:hover fieldset': {
                borderColor: isDarkMode ? '#606060' : '#c0c0c0',
              },
              '&.Mui-focused fieldset': {
                borderColor: '#1976d2',
              },
            },
          },
        },
      },
      MuiAppBar: {
        styleOverrides: {
          root: {
            backgroundColor: isDarkMode ? '#1e1e1e' : '#ffffff',
            color: isDarkMode ? '#ffffff' : '#212121',
            boxShadow: isDarkMode 
              ? '0 2px 4px rgba(0,0,0,0.3)' 
              : '0 2px 4px rgba(0,0,0,0.1)',
          },
        },
      },
      MuiDrawer: {
        styleOverrides: {
          paper: {
            backgroundColor: isDarkMode ? '#1e1e1e' : '#ffffff',
            borderRight: `1px solid ${isDarkMode ? '#404040' : '#e0e0e0'}`,
          },
        },
      },
      MuiListItem: {
        styleOverrides: {
          root: {
            '&:hover': {
              backgroundColor: isDarkMode ? '#2a2a2a' : '#f5f5f5',
            },
            '&.Mui-selected': {
              backgroundColor: isDarkMode ? '#1976d2' : '#e3f2fd',
              '&:hover': {
                backgroundColor: isDarkMode ? '#1565c0' : '#bbdefb',
              },
            },
          },
        },
      },
      MuiChip: {
        styleOverrides: {
          root: {
            backgroundColor: isDarkMode ? '#2a2a2a' : '#f5f5f5',
            color: isDarkMode ? '#ffffff' : '#212121',
          },
        },
      },
      MuiTableCell: {
        styleOverrides: {
          root: {
            borderBottom: `1px solid ${isDarkMode ? '#404040' : '#e0e0e0'}`,
          },
          head: {
            backgroundColor: isDarkMode ? '#2a2a2a' : '#f5f5f5',
            fontWeight: 600,
          },
        },
      },
    },
  });

  const toggleTheme = () => {
    const newMode = mode === 'light' ? 'dark' : 'light';
    setThemeMode(newMode);
  };

  const setThemeMode = (newMode: ThemeMode) => {
    setMode(newMode);
    localStorage.setItem('theme_mode', newMode);
  };

  const value: ThemeContextType = {
    mode,
    theme,
    toggleTheme,
    setThemeMode,
    isDarkMode,
  };

  return (
    <ThemeContext.Provider value={value}>
      <ThemeProvider theme={theme}>
        {children}
      </ThemeProvider>
    </ThemeContext.Provider>
  );
};

export default ThemeContext;