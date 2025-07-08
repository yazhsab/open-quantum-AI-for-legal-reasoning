"""
Quantum Attention Mechanism

This module implements quantum attention mechanisms for legal reasoning,
replacing classical attention with quantum interference patterns to
process multiple legal precedents simultaneously.

Copyright 2024 XQELM Research Team
Licensed under the Apache License, Version 2.0
"""

import asyncio
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
from dataclasses import dataclass
from loguru import logger

import pennylane as qml
from pennylane import numpy as qnp
import torch


@dataclass
class QuantumAttentionResult:
    """Result of quantum attention computation."""
    attention_weights: qnp.ndarray
    attended_states: List[qnp.ndarray]
    interference_pattern: qnp.ndarray
    coherence_measure: float
    attention_entropy: float
    head_outputs: List[qnp.ndarray]
    metadata: Dict[str, Any]


class QuantumAttentionMechanism:
    """
    Quantum Attention Mechanism (QAM) for legal reasoning.
    
    This class implements quantum attention using interference patterns
    and quantum superposition to process multiple legal concepts simultaneously.
    """
    
    def __init__(
        self,
        n_qubits: int = 12,
        n_heads: int = 4,
        n_layers: int = 2,
        device: Optional[qml.Device] = None
    ):
        """
        Initialize the Quantum Attention Mechanism.
        
        Args:
            n_qubits: Number of qubits for attention circuits
            n_heads: Number of attention heads
            n_layers: Number of attention layers
            device: PennyLane quantum device
        """
        self.n_qubits = n_qubits
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.device = device or qml.device("default.qubit", wires=n_qubits)
        
        # Ensure we have enough qubits for multi-head attention
        self.qubits_per_head = max(2, n_qubits // n_heads)
        
        # Initialize parameters
        self._initialize_parameters()
        
        # Attention type configurations
        self.attention_configs = {
            'precedent': {'interference_strength': 0.8, 'coherence_threshold': 0.6},
            'statute': {'interference_strength': 0.9, 'coherence_threshold': 0.8},
            'principle': {'interference_strength': 0.7, 'coherence_threshold': 0.5},
            'case_fact': {'interference_strength': 0.6, 'coherence_threshold': 0.4}
        }
        
        logger.info(f"Initialized QuantumAttentionMechanism with {n_heads} heads")
    
    def _initialize_parameters(self) -> None:
        """Initialize quantum attention parameters."""
        # Query transformation parameters
        self.query_params = qnp.random.normal(
            0, 0.1, (self.n_layers, self.qubits_per_head, 3)
        )
        
        # Key transformation parameters
        self.key_params = qnp.random.normal(
            0, 0.1, (self.n_layers, self.qubits_per_head, 3)
        )
        
        # Value transformation parameters
        self.value_params = qnp.random.normal(
            0, 0.1, (self.n_layers, self.qubits_per_head, 3)
        )
        
        # Interference parameters for quantum attention
        self.interference_params = qnp.random.normal(
            0, 0.1, (self.n_heads, self.qubits_per_head)
        )
        
        # Multi-head combination parameters
        self.combination_params = qnp.random.normal(
            0, 0.1, (self.n_heads, self.qubits_per_head)
        )
    
    @qml.qnode(device=None)
    def _quantum_attention_circuit(
        self,
        query_state: qnp.ndarray,
        key_states: List[qnp.ndarray],
        value_states: List[qnp.ndarray],
        head_id: int,
        query_params: qnp.ndarray,
        key_params: qnp.ndarray,
        value_params: qnp.ndarray,
        interference_params: qnp.ndarray
    ) -> qnp.ndarray:
        """
        Quantum circuit for attention computation.
        
        Args:
            query_state: Query quantum state
            key_states: List of key quantum states
            value_states: List of value quantum states
            head_id: Attention head identifier
            query_params: Query transformation parameters
            key_params: Key transformation parameters
            value_params: Value transformation parameters
            interference_params: Interference parameters
            
        Returns:
            Attended quantum state
        """
        n_keys = len(key_states)
        qubits_start = head_id * self.qubits_per_head
        qubits_end = qubits_start + self.qubits_per_head
        head_qubits = list(range(qubits_start, min(qubits_end, self.n_qubits)))
        
        if len(head_qubits) < 2:
            # Fallback for insufficient qubits
            head_qubits = [0, 1]
        
        # Step 1: Prepare query state
        self._prepare_quantum_state(query_state, head_qubits[:len(head_qubits)//2])
        
        # Step 2: Apply query transformation
        for layer in range(self.n_layers):
            for i, qubit in enumerate(head_qubits[:len(head_qubits)//2]):
                if i < len(query_params[layer]):
                    qml.RX(query_params[layer, i, 0], wires=qubit)
                    qml.RY(query_params[layer, i, 1], wires=qubit)
                    qml.RZ(query_params[layer, i, 2], wires=qubit)
        
        # Step 3: Prepare key-value superposition
        key_qubits = head_qubits[len(head_qubits)//2:]
        if key_qubits and n_keys > 0:
            # Create superposition of keys
            for i, qubit in enumerate(key_qubits):
                qml.Hadamard(wires=qubit)
            
            # Encode keys using controlled operations
            for key_idx, (key_state, value_state) in enumerate(zip(key_states, value_states)):
                if key_idx < len(key_qubits):
                    control_qubit = key_qubits[key_idx]
                    
                    # Apply key transformation
                    for layer in range(self.n_layers):
                        if key_idx < len(key_params[layer]):
                            qml.CRX(key_params[layer, key_idx, 0], 
                                   wires=[control_qubit, head_qubits[0]])
                            qml.CRY(key_params[layer, key_idx, 1], 
                                   wires=[control_qubit, head_qubits[0]])
                            qml.CRZ(key_params[layer, key_idx, 2], 
                                   wires=[control_qubit, head_qubits[0]])
        
        # Step 4: Quantum interference for attention
        self._apply_quantum_interference(head_qubits, interference_params[head_id])
        
        # Step 5: Attention weight extraction through measurement
        # Use amplitude amplification for relevant states
        self._amplitude_amplification(head_qubits)
        
        return qml.state()
    
    def _prepare_quantum_state(
        self,
        classical_state: qnp.ndarray,
        qubits: List[int]
    ) -> None:
        """Prepare quantum state from classical representation."""
        if len(qubits) == 0:
            return
        
        # Normalize classical state
        norm = qnp.linalg.norm(classical_state)
        if norm > 0:
            normalized_state = classical_state / norm
        else:
            normalized_state = qnp.ones(len(classical_state)) / qnp.sqrt(len(classical_state))
        
        # Encode into quantum state using amplitude encoding
        n_amplitudes = 2 ** min(len(qubits), int(np.log2(len(normalized_state))))
        
        if n_amplitudes > 1:
            amplitudes = normalized_state[:n_amplitudes]
            # Renormalize
            amplitudes = amplitudes / qnp.linalg.norm(amplitudes)
            
            try:
                qml.AmplitudeEmbedding(
                    features=amplitudes,
                    wires=qubits[:int(np.log2(n_amplitudes))],
                    normalize=True
                )
            except:
                # Fallback: simple rotation encoding
                for i, qubit in enumerate(qubits):
                    if i < len(normalized_state):
                        angle = normalized_state[i] * np.pi
                        qml.RY(angle, wires=qubit)
    
    def _apply_quantum_interference(
        self,
        qubits: List[int],
        interference_params: qnp.ndarray
    ) -> None:
        """Apply quantum interference patterns for attention."""
        n_qubits = len(qubits)
        
        if n_qubits < 2:
            return
        
        # Create interference patterns using controlled rotations
        for i in range(n_qubits - 1):
            if i < len(interference_params):
                # Controlled phase gates for interference
                qml.CPhase(interference_params[i], wires=[qubits[i], qubits[i + 1]])
                
                # Additional interference through controlled rotations
                qml.CRY(interference_params[i] * 0.5, wires=[qubits[i], qubits[i + 1]])
        
        # Ring connectivity for global interference
        if n_qubits > 2 and len(interference_params) > n_qubits - 1:
            qml.CPhase(interference_params[-1], wires=[qubits[-1], qubits[0]])
    
    def _amplitude_amplification(self, qubits: List[int]) -> None:
        """Apply amplitude amplification for attention weight enhancement."""
        if len(qubits) < 2:
            return
        
        # Simple amplitude amplification using Grover-like operations
        # Oracle: mark states with high attention
        for qubit in qubits:
            qml.PauliZ(wires=qubit)
        
        # Diffusion operator
        for qubit in qubits:
            qml.Hadamard(wires=qubit)
            qml.PauliZ(wires=qubit)
            qml.Hadamard(wires=qubit)
    
    async def compute_attention(
        self,
        query_embedding: qnp.ndarray,
        key_value_embeddings: List[qnp.ndarray],
        attention_type: str = 'precedent'
    ) -> QuantumAttentionResult:
        """
        Compute quantum attention weights and attended representations.
        
        Args:
            query_embedding: Query quantum state or classical embedding
            key_value_embeddings: List of key-value embeddings
            attention_type: Type of attention (precedent, statute, etc.)
            
        Returns:
            QuantumAttentionResult with attention weights and states
        """
        if not key_value_embeddings:
            # Return empty result for no keys
            return QuantumAttentionResult(
                attention_weights=qnp.array([]),
                attended_states=[],
                interference_pattern=qnp.array([]),
                coherence_measure=0.0,
                attention_entropy=0.0,
                head_outputs=[],
                metadata={'attention_type': attention_type, 'n_keys': 0}
            )
        
        # Get attention configuration
        config = self.attention_configs.get(attention_type, self.attention_configs['precedent'])
        
        # Prepare key and value states
        key_states = key_value_embeddings
        value_states = key_value_embeddings  # In this implementation, keys = values
        
        # Multi-head attention computation
        head_outputs = []
        attention_weights_per_head = []
        
        for head_id in range(self.n_heads):
            # Create quantum circuit for this head
            circuit = qml.QNode(
                self._quantum_attention_circuit,
                device=self.device,
                interface="numpy"
            )
            
            # Execute attention circuit
            attended_state = circuit(
                query_embedding,
                key_states,
                value_states,
                head_id,
                self.query_params,
                self.key_params,
                self.value_params,
                self.interference_params
            )
            
            head_outputs.append(attended_state)
            
            # Extract attention weights from quantum state
            weights = self._extract_attention_weights(
                attended_state, len(key_states), head_id
            )
            attention_weights_per_head.append(weights)
        
        # Combine multi-head outputs
        combined_weights = self._combine_attention_heads(attention_weights_per_head)
        combined_state = self._combine_head_states(head_outputs)
        
        # Calculate interference pattern
        interference_pattern = self._calculate_interference_pattern(head_outputs)
        
        # Calculate coherence measure
        coherence = self._calculate_coherence(combined_state, config['coherence_threshold'])
        
        # Calculate attention entropy
        entropy = self._calculate_attention_entropy(combined_weights)
        
        # Create attended states
        attended_states = self._create_attended_states(
            value_states, combined_weights
        )
        
        result = QuantumAttentionResult(
            attention_weights=combined_weights,
            attended_states=attended_states,
            interference_pattern=interference_pattern,
            coherence_measure=coherence,
            attention_entropy=entropy,
            head_outputs=head_outputs,
            metadata={
                'attention_type': attention_type,
                'n_keys': len(key_states),
                'n_heads': self.n_heads,
                'config': config
            }
        )
        
        logger.debug(f"Computed {attention_type} attention with coherence: {coherence:.3f}")
        return result
    
    def _extract_attention_weights(
        self,
        quantum_state: qnp.ndarray,
        n_keys: int,
        head_id: int
    ) -> qnp.ndarray:
        """Extract attention weights from quantum state."""
        # Get amplitudes corresponding to attention weights
        amplitudes = qnp.abs(quantum_state) ** 2
        
        # Extract weights for this head's qubits
        qubits_start = head_id * self.qubits_per_head
        qubits_end = qubits_start + self.qubits_per_head
        
        # Map quantum amplitudes to attention weights
        n_states = min(n_keys, 2 ** self.qubits_per_head)
        weights = qnp.zeros(n_keys)
        
        for i in range(min(n_keys, len(amplitudes))):
            weights[i] = amplitudes[i]
        
        # Normalize weights
        weight_sum = qnp.sum(weights)
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            weights = qnp.ones(n_keys) / n_keys
        
        return weights
    
    def _combine_attention_heads(
        self,
        head_weights: List[qnp.ndarray]
    ) -> qnp.ndarray:
        """Combine attention weights from multiple heads."""
        if not head_weights:
            return qnp.array([])
        
        # Average across heads with learned combination weights
        combined = qnp.zeros_like(head_weights[0])
        
        for i, weights in enumerate(head_weights):
            if i < len(self.combination_params):
                head_weight = qnp.mean(qnp.abs(self.combination_params[i]))
                combined += head_weight * weights
        
        # Normalize
        weight_sum = qnp.sum(combined)
        if weight_sum > 0:
            combined = combined / weight_sum
        
        return combined
    
    def _combine_head_states(
        self,
        head_states: List[qnp.ndarray]
    ) -> qnp.ndarray:
        """Combine quantum states from multiple attention heads."""
        if not head_states:
            return qnp.array([])
        
        # Weighted combination of head states
        combined_state = qnp.zeros_like(head_states[0])
        
        for i, state in enumerate(head_states):
            if i < len(self.combination_params):
                weight = qnp.mean(qnp.abs(self.combination_params[i]))
                combined_state += weight * state
        
        # Normalize
        norm = qnp.linalg.norm(combined_state)
        if norm > 0:
            combined_state = combined_state / norm
        
        return combined_state
    
    def _calculate_interference_pattern(
        self,
        head_states: List[qnp.ndarray]
    ) -> qnp.ndarray:
        """Calculate quantum interference pattern between attention heads."""
        if len(head_states) < 2:
            return qnp.array([0.0])
        
        interference = qnp.zeros(len(head_states[0]))
        
        # Calculate pairwise interference
        for i in range(len(head_states)):
            for j in range(i + 1, len(head_states)):
                # Quantum interference term
                interference += qnp.real(
                    qnp.conj(head_states[i]) * head_states[j]
                )
        
        return interference
    
    def _calculate_coherence(
        self,
        quantum_state: qnp.ndarray,
        threshold: float
    ) -> float:
        """Calculate quantum coherence measure."""
        # Calculate l1-norm of coherence
        density_matrix = qnp.outer(quantum_state, qnp.conj(quantum_state))
        
        # Remove diagonal elements
        coherence_matrix = density_matrix - qnp.diag(qnp.diag(density_matrix))
        
        # L1-norm coherence
        coherence = qnp.sum(qnp.abs(coherence_matrix))
        
        return float(coherence)
    
    def _calculate_attention_entropy(self, attention_weights: qnp.ndarray) -> float:
        """Calculate entropy of attention distribution."""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        weights = attention_weights + epsilon
        
        # Normalize
        weights = weights / qnp.sum(weights)
        
        # Calculate Shannon entropy
        entropy = -qnp.sum(weights * qnp.log(weights))
        
        return float(entropy)
    
    def _create_attended_states(
        self,
        value_states: List[qnp.ndarray],
        attention_weights: qnp.ndarray
    ) -> List[qnp.ndarray]:
        """Create attended value states using attention weights."""
        attended_states = []
        
        for i, (value_state, weight) in enumerate(zip(value_states, attention_weights)):
            # Weight the value state by attention
            attended_state = weight * value_state
            attended_states.append(attended_state)
        
        return attended_states
    
    async def compute_self_attention(
        self,
        embeddings: List[qnp.ndarray],
        attention_type: str = 'precedent'
    ) -> QuantumAttentionResult:
        """
        Compute self-attention over a sequence of embeddings.
        
        Args:
            embeddings: List of quantum embeddings
            attention_type: Type of attention computation
            
        Returns:
            QuantumAttentionResult with self-attention weights
        """
        if not embeddings:
            return QuantumAttentionResult(
                attention_weights=qnp.array([]),
                attended_states=[],
                interference_pattern=qnp.array([]),
                coherence_measure=0.0,
                attention_entropy=0.0,
                head_outputs=[],
                metadata={'attention_type': attention_type, 'self_attention': True}
            )
        
        # For self-attention, each embedding attends to all others
        all_results = []
        
        for i, query_embedding in enumerate(embeddings):
            # Use all embeddings as keys/values
            result = await self.compute_attention(
                query_embedding,
                embeddings,
                attention_type
            )
            all_results.append(result)
        
        # Combine results
        combined_weights = qnp.mean([r.attention_weights for r in all_results], axis=0)
        combined_states = [r.attended_states for r in all_results]
        
        return QuantumAttentionResult(
            attention_weights=combined_weights,
            attended_states=combined_states,
            interference_pattern=qnp.mean([r.interference_pattern for r in all_results], axis=0),
            coherence_measure=qnp.mean([r.coherence_measure for r in all_results]),
            attention_entropy=qnp.mean([r.attention_entropy for r in all_results]),
            head_outputs=[r.head_outputs for r in all_results],
            metadata={'attention_type': attention_type, 'self_attention': True}
        )
    
    def update_parameters(self, gradients: Dict[str, qnp.ndarray]) -> None:
        """Update attention parameters based on gradients."""
        learning_rate = 0.01
        
        if 'query_params' in gradients:
            self.query_params -= learning_rate * gradients['query_params']
        
        if 'key_params' in gradients:
            self.key_params -= learning_rate * gradients['key_params']
        
        if 'value_params' in gradients:
            self.value_params -= learning_rate * gradients['value_params']
        
        if 'interference_params' in gradients:
            self.interference_params -= learning_rate * gradients['interference_params']
        
        if 'combination_params' in gradients:
            self.combination_params -= learning_rate * gradients['combination_params']
    
    def get_parameters(self) -> Dict[str, qnp.ndarray]:
        """Get current attention parameters."""
        return {
            'query_params': self.query_params,
            'key_params': self.key_params,
            'value_params': self.value_params,
            'interference_params': self.interference_params,
            'combination_params': self.combination_params
        }
    
    def set_parameters(self, parameters: Dict[str, qnp.ndarray]) -> None:
        """Set attention parameters."""
        for param_name, param_value in parameters.items():
            if hasattr(self, param_name):
                setattr(self, param_name, param_value)