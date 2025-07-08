# Contributing to XQELM

Thank you for your interest in contributing to the Explainable Quantum-Enhanced Language Models for Legal Reasoning (XQELM) project! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

### Our Pledge

We are committed to making participation in this project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Node.js 16 or higher (for frontend development)
- Docker and Docker Compose
- Git
- Basic understanding of quantum computing concepts
- Familiarity with legal terminology (helpful but not required)

### Development Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/your-username/explainable-quantum-enhanced-language-models-legal-reasoning.git
   cd explainable-quantum-enhanced-language-models-legal-reasoning
   ```

2. **Set Up Python Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev,test]"
   ```

3. **Set Up Frontend Environment**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

4. **Set Up Development Databases**
   ```bash
   docker-compose -f docker-compose.dev.yml up -d postgres redis neo4j
   ```

5. **Initialize Database**
   ```bash
   python -m src.xqelm.database.init_db
   ```

6. **Run Tests**
   ```bash
   pytest tests/
   cd frontend && npm test
   ```

## Contributing Guidelines

### Types of Contributions

We welcome various types of contributions:

- **Bug Reports**: Help us identify and fix issues
- **Feature Requests**: Suggest new functionality
- **Code Contributions**: Implement features, fix bugs, improve performance
- **Documentation**: Improve docs, add examples, write tutorials
- **Research**: Contribute to quantum algorithms, legal reasoning methods
- **Testing**: Add test cases, improve test coverage
- **UI/UX**: Improve frontend design and user experience

### Areas of Focus

1. **Quantum Computing**
   - Quantum circuit optimization
   - New quantum algorithms for legal reasoning
   - Quantum machine learning improvements
   - Error mitigation techniques

2. **Legal AI**
   - Legal text processing
   - Case law analysis
   - Statutory interpretation
   - Legal entity recognition

3. **Explainable AI**
   - Quantum state visualization
   - Decision explanation methods
   - Interpretability improvements
   - User-friendly explanations

4. **System Architecture**
   - Performance optimization
   - Scalability improvements
   - Security enhancements
   - Infrastructure automation

5. **Frontend Development**
   - User interface improvements
   - Data visualization
   - Accessibility features
   - Mobile responsiveness

## Pull Request Process

### Before Submitting

1. **Check Existing Issues**: Look for related issues or discussions
2. **Create an Issue**: For significant changes, create an issue first
3. **Fork the Repository**: Work on your own fork
4. **Create a Branch**: Use descriptive branch names
   ```bash
   git checkout -b feature/quantum-attention-mechanism
   git checkout -b fix/authentication-bug
   git checkout -b docs/api-documentation
   ```

### Development Process

1. **Write Code**
   - Follow coding standards (see below)
   - Add appropriate tests
   - Update documentation

2. **Test Your Changes**
   ```bash
   # Run all tests
   pytest tests/
   
   # Run specific test categories
   pytest tests/ -m "unit"
   pytest tests/ -m "integration"
   pytest tests/ -m "quantum"
   
   # Run frontend tests
   cd frontend && npm test
   
   # Run linting
   flake8 src tests
   mypy src
   cd frontend && npm run lint
   ```

3. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add quantum attention mechanism for legal reasoning"
   ```

### Commit Message Format

We use conventional commits for clear and consistent commit messages:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(quantum): add quantum attention mechanism
fix(api): resolve authentication token expiration
docs(readme): update installation instructions
test(quantum): add unit tests for quantum circuits
```

### Submitting the Pull Request

1. **Push to Your Fork**
   ```bash
   git push origin feature/quantum-attention-mechanism
   ```

2. **Create Pull Request**
   - Use a clear, descriptive title
   - Fill out the PR template completely
   - Link related issues
   - Add screenshots for UI changes

3. **PR Template**
   ```markdown
   ## Description
   Brief description of changes

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update

   ## Testing
   - [ ] Unit tests pass
   - [ ] Integration tests pass
   - [ ] Manual testing completed

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] Tests added/updated
   ```

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

1. **Clear Title**: Descriptive summary of the issue
2. **Environment**: OS, Python version, dependencies
3. **Steps to Reproduce**: Detailed steps to recreate the bug
4. **Expected Behavior**: What should happen
5. **Actual Behavior**: What actually happens
6. **Screenshots/Logs**: If applicable
7. **Additional Context**: Any other relevant information

### Feature Requests

For feature requests, please include:

1. **Problem Statement**: What problem does this solve?
2. **Proposed Solution**: How should it work?
3. **Alternatives**: Other solutions considered
4. **Use Cases**: Real-world scenarios
5. **Implementation Ideas**: Technical approach (if any)

## Development Workflow

### Coding Standards

#### Python Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Maximum line length: 88 characters (Black formatter)
- Use docstrings for all public functions and classes
- Import organization: standard library, third-party, local imports

```python
"""
Module docstring describing the purpose.
"""

from typing import List, Dict, Optional
import asyncio

import numpy as np
import pennylane as qml

from ..utils.config import Config


class QuantumCircuit:
    """
    Quantum circuit for legal reasoning.
    
    Args:
        num_qubits: Number of qubits in the circuit
        backend: Quantum backend to use
    """
    
    def __init__(self, num_qubits: int, backend: str = "default.qubit") -> None:
        self.num_qubits = num_qubits
        self.backend = backend
    
    async def execute(self, parameters: np.ndarray) -> Dict[str, float]:
        """
        Execute the quantum circuit.
        
        Args:
            parameters: Circuit parameters
            
        Returns:
            Measurement results
        """
        # Implementation here
        pass
```

#### TypeScript/React Code Style

- Use TypeScript for all new code
- Follow React best practices
- Use functional components with hooks
- Implement proper error boundaries
- Use Material-UI components consistently

```typescript
interface QuantumVisualizationProps {
  quantumState: number[];
  circuitDepth: number;
  onStateChange?: (state: number[]) => void;
}

const QuantumVisualization: React.FC<QuantumVisualizationProps> = ({
  quantumState,
  circuitDepth,
  onStateChange
}) => {
  const [isLoading, setIsLoading] = useState(false);
  
  useEffect(() => {
    // Effect implementation
  }, [quantumState]);
  
  return (
    <Box>
      {/* Component JSX */}
    </Box>
  );
};

export default QuantumVisualization;
```

### Documentation Standards

- Use clear, concise language
- Include code examples
- Add diagrams for complex concepts
- Keep documentation up-to-date with code changes
- Use proper markdown formatting

### Testing Standards

#### Unit Tests
- Test individual functions and classes
- Mock external dependencies
- Aim for >90% code coverage
- Use descriptive test names

```python
@pytest.mark.unit
async def test_quantum_circuit_execution_with_valid_parameters():
    """Test quantum circuit execution with valid parameters."""
    circuit = QuantumCircuit(num_qubits=4)
    parameters = np.random.random(8)
    
    result = await circuit.execute(parameters)
    
    assert isinstance(result, dict)
    assert "expectation_value" in result
    assert 0 <= result["expectation_value"] <= 1
```

#### Integration Tests
- Test component interactions
- Use test databases
- Test API endpoints end-to-end

#### Quantum Tests
- Test quantum algorithms
- Verify quantum state properties
- Test classical-quantum interfaces

## Documentation

### Types of Documentation

1. **API Documentation**: Auto-generated from docstrings
2. **User Guides**: Step-by-step tutorials
3. **Developer Guides**: Technical implementation details
4. **Research Papers**: Academic documentation
5. **Architecture Docs**: System design and decisions

### Writing Guidelines

- **Audience**: Consider who will read the documentation
- **Structure**: Use clear headings and organization
- **Examples**: Include practical code examples
- **Diagrams**: Use visual aids for complex concepts
- **Updates**: Keep documentation current with code

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs
make html

# Serve documentation locally
python -m http.server 8080 -d _build/html
```

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Discord**: Real-time chat (link in README)
- **Email**: maintainers@xqelm.org

### Getting Help

1. **Check Documentation**: Look for existing answers
2. **Search Issues**: See if someone else had the same problem
3. **Ask Questions**: Use GitHub Discussions for general questions
4. **Join Community**: Connect with other contributors

### Mentorship

We welcome new contributors and provide mentorship:

- **Good First Issues**: Labeled for newcomers
- **Mentorship Program**: Pair new contributors with experienced ones
- **Code Reviews**: Learn through feedback
- **Office Hours**: Regular Q&A sessions

## Recognition

We value all contributions and recognize contributors through:

- **Contributors List**: Listed in README and documentation
- **Release Notes**: Contributions highlighted in releases
- **Community Spotlight**: Featured contributors
- **Conference Presentations**: Opportunities to present work

## Legal

### Licensing

By contributing to XQELM, you agree that your contributions will be licensed under the Apache License 2.0.

### Copyright

- Retain copyright to your contributions
- Grant XQELM project rights to use contributions
- Ensure you have rights to contribute code

### Patents

Contributors grant a patent license for their contributions as outlined in the Apache License 2.0.

## Questions?

If you have questions about contributing, please:

1. Check this document first
2. Search existing issues and discussions
3. Create a new discussion or issue
4. Contact maintainers directly

Thank you for contributing to XQELM! Your efforts help advance the intersection of quantum computing and legal AI.