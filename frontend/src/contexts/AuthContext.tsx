import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import axios from 'axios';
import toast from 'react-hot-toast';

// Types
interface User {
  id: string;
  username: string;
  email: string;
  full_name?: string;
  organization?: string;
  role: string;
  is_active: boolean;
  is_verified: boolean;
  rate_limit_tier: string;
  total_queries: number;
  total_tokens_used: number;
  last_login?: string;
  created_at: string;
  updated_at: string;
}

interface AuthContextType {
  user: User | null;
  token: string | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (username: string, password: string) => Promise<boolean>;
  register: (userData: RegisterData) => Promise<boolean>;
  logout: () => void;
  refreshUser: () => Promise<void>;
  updateProfile: (userData: Partial<User>) => Promise<boolean>;
}

interface RegisterData {
  username: string;
  email: string;
  password: string;
  full_name?: string;
  organization?: string;
}

interface LoginResponse {
  access_token: string;
  token_type: string;
  user: User;
}

// Create context
const AuthContext = createContext<AuthContextType | undefined>(undefined);

// Custom hook to use auth context
export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

// Auth provider component
interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Configure axios defaults
  useEffect(() => {
    const storedToken = localStorage.getItem('auth_token');
    if (storedToken) {
      setToken(storedToken);
      axios.defaults.headers.common['Authorization'] = `Bearer ${storedToken}`;
    }
    
    // Set base URL for API requests
    axios.defaults.baseURL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
    
    // Add response interceptor to handle token expiration
    axios.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          // Token expired or invalid
          logout();
          toast.error('Session expired. Please login again.');
        }
        return Promise.reject(error);
      }
    );
    
    setIsLoading(false);
  }, []);

  // Check if user is authenticated on mount
  useEffect(() => {
    if (token && !user) {
      refreshUser();
    }
  }, [token, user]);

  const login = async (username: string, password: string): Promise<boolean> => {
    try {
      setIsLoading(true);
      
      const formData = new FormData();
      formData.append('username', username);
      formData.append('password', password);
      
      const response = await axios.post<LoginResponse>('/auth/login', formData, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
      });
      
      const { access_token, user: userData } = response.data;
      
      // Store token and user data
      setToken(access_token);
      setUser(userData);
      localStorage.setItem('auth_token', access_token);
      axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
      
      toast.success('Login successful!');
      return true;
      
    } catch (error: any) {
      console.error('Login error:', error);
      const message = error.response?.data?.detail || 'Login failed. Please try again.';
      toast.error(message);
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  const register = async (userData: RegisterData): Promise<boolean> => {
    try {
      setIsLoading(true);
      
      const response = await axios.post('/auth/register', userData);
      
      toast.success('Registration successful! Please login with your credentials.');
      return true;
      
    } catch (error: any) {
      console.error('Registration error:', error);
      const message = error.response?.data?.detail || 'Registration failed. Please try again.';
      toast.error(message);
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  const logout = (): void => {
    setUser(null);
    setToken(null);
    localStorage.removeItem('auth_token');
    delete axios.defaults.headers.common['Authorization'];
    toast.success('Logged out successfully');
  };

  const refreshUser = async (): Promise<void> => {
    try {
      if (!token) return;
      
      const response = await axios.get<User>('/auth/me');
      setUser(response.data);
      
    } catch (error: any) {
      console.error('Failed to refresh user data:', error);
      if (error.response?.status === 401) {
        logout();
      }
    }
  };

  const updateProfile = async (userData: Partial<User>): Promise<boolean> => {
    try {
      setIsLoading(true);
      
      const response = await axios.put<User>('/auth/profile', userData);
      setUser(response.data);
      
      toast.success('Profile updated successfully!');
      return true;
      
    } catch (error: any) {
      console.error('Profile update error:', error);
      const message = error.response?.data?.detail || 'Failed to update profile. Please try again.';
      toast.error(message);
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  const isAuthenticated = !!user && !!token;

  const value: AuthContextType = {
    user,
    token,
    isLoading,
    isAuthenticated,
    login,
    register,
    logout,
    refreshUser,
    updateProfile,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

// Higher-order component for protected routes
export const withAuth = <P extends object>(
  Component: React.ComponentType<P>
): React.FC<P> => {
  return (props: P) => {
    const { isAuthenticated, isLoading } = useAuth();
    
    if (isLoading) {
      return (
        <div className="flex items-center justify-center min-h-screen">
          <div className="loading-spinner"></div>
        </div>
      );
    }
    
    if (!isAuthenticated) {
      return (
        <div className="flex items-center justify-center min-h-screen">
          <div className="text-center">
            <h2>Access Denied</h2>
            <p>Please login to access this page.</p>
          </div>
        </div>
      );
    }
    
    return <Component {...props} />;
  };
};

export default AuthContext;