"""
Quantum Legal Embeddings

This module implements quantum embeddings for legal concepts using PennyLane.
It encodes legal concepts into quantum superposition states while preserving
semantic relationships and enabling quantum interference patterns.

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
import torch.nn as nn


@dataclass
class QuantumEmbeddingResult:
    """Result of quantum embedding operation."""
    quantum_state: qnp.ndarray
    amplitudes: qnp.ndarray
    phases: qnp.ndarray
    entanglement_measure: float
    fidelity: float
    concept_type: str
    metadata: Dict[str, Any]


class QuantumLegalEmbedding:
    """
    Quantum Legal Embedding (QLE) module for encoding legal concepts
    into quantum superposition states.
    
    This class implements amplitude encoding and variational quantum embeddings
    specifically designed for legal domain concepts.
    """
    
    def __init__(
        self,
        n_qubits: int = 10,
        n_layers: int = 3,
        device: Optional[qml.Device] = None,
        embedding_dim: int = 512
    ):
        """
        Initialize the Quantum Legal Embedding module.
        
        Args:
            n_qubits: Number of qubits for quantum circuits
            n_layers: Number of variational layers
            device: PennyLane quantum device
            embedding_dim: Dimension of classical embeddings
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.device = device or qml.device("default.qubit", wires=n_qubits)
        
        # Initialize parameters
        self._initialize_parameters()
        
        # Legal concept type mappings
        self.concept_types = {
            'query': 0,
            'precedent': 1,
            'statute': 2,
            'principle': 3,
            'case_fact': 4,
            'legal_entity': 5
        }
        
        logger.info(f"Initialized QuantumLegalEmbedding with {n_qubits} qubits")
    
    def _initialize_parameters(self) -> None:
        """Initialize quantum circuit parameters."""
        # Variational parameters for embedding layers
        self.embedding_params = qnp.random.normal(
            0, 0.1, (self.n_layers, self.n_qubits, 3)
        )
        
        # Entangling parameters
        self.entangling_params = qnp.random.normal(
            0, 0.1, (self.n_layers, self.n_qubits - 1)
        )
        
        # Concept-specific parameters
        self.concept_params = qnp.random.normal(
            0, 0.1, (len(self.concept_types), self.n_qubits)
        )
    
    @qml.qnode(device=None)
    def _quantum_embedding_circuit(
        self,
        classical_features: qnp.ndarray,
        concept_type_id: int,
        embedding_params: qnp.ndarray,
        entangling_params: qnp.ndarray,
        concept_params: qnp.ndarray
    ) -> qnp.ndarray:
        """
        Quantum circuit for legal concept embedding.
        
        Args:
            classical_features: Normalized classical feature vector
            concept_type_id: ID of the legal concept type
            embedding_params: Variational embedding parameters
            entangling_params: Entangling gate parameters
            concept_params: Concept-specific parameters
            
        Returns:
            Quantum state amplitudes
        """
        n_features = len(classical_features)
        n_qubits_for_features = min(self.n_qubits - 1, n_features)
        
        # Step 1: Amplitude encoding of classical features
        # Normalize features to valid quantum amplitudes
        normalized_features = self._normalize_to_amplitudes(
            classical_features[:n_qubits_for_features]
        )
        
        # Initialize state with amplitude encoding
        qml.AmplitudeEmbedding(
            features=normalized_features,
            wires=range(n_qubits_for_features),
            normalize=True
        )
        
        # Step 2: Concept type encoding
        concept_qubit = self.n_qubits - 1
        if concept_type_id % 2 == 1:
            qml.PauliX(wires=concept_qubit)
        
        # Apply concept-specific rotation
        qml.RY(concept_params[concept_type_id, concept_qubit], wires=concept_qubit)
        
        # Step 3: Variational embedding layers
        for layer in range(self.n_layers):
            # Single-qubit rotations
            for qubit in range(self.n_qubits):
                qml.RX(embedding_params[layer, qubit, 0], wires=qubit)
                qml.RY(embedding_params[layer, qubit, 1], wires=qubit)
                qml.RZ(embedding_params[layer, qubit, 2], wires=qubit)
            
            # Entangling layer
            for qubit in range(self.n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])
                qml.RZ(entangling_params[layer, qubit], wires=qubit + 1)
            
            # Ring connectivity for better entanglement
            if self.n_qubits > 2:
                qml.CNOT(wires=[self.n_qubits - 1, 0])
        
        # Step 4: Legal domain-specific gates
        self._apply_legal_domain_gates(concept_type_id)
        
        # Return state vector
        return qml.state()
    
    def _apply_legal_domain_gates(self, concept_type_id: int) -> None:
        """Apply legal domain-specific quantum gates."""
        
        if concept_type_id == self.concept_types['precedent']:
            # Precedent concepts get additional entanglement
            for i in range(0, self.n_qubits - 1, 2):
                qml.CZ(wires=[i, i + 1])
        
        elif concept_type_id == self.concept_types['statute']:
            # Statutory concepts get phase gates for rigidity
            for i in range(self.n_qubits):
                qml.PhaseShift(np.pi / 4, wires=i)
        
        elif concept_type_id == self.concept_types['principle']:
            # Legal principles get Hadamard gates for superposition
            for i in range(min(3, self.n_qubits)):
                qml.Hadamard(wires=i)
    
    def _normalize_to_amplitudes(self, features: qnp.ndarray) -> qnp.ndarray:
        """Normalize features to valid quantum amplitudes."""
        # Ensure we have the right number of features for amplitude encoding
        n_amplitudes = 2 ** min(int(np.log2(len(features))), self.n_qubits - 1)
        
        if len(features) < n_amplitudes:
            # Pad with zeros
            padded_features = qnp.zeros(n_amplitudes)
            padded_features[:len(features)] = features
            features = padded_features
        else:
            # Truncate
            features = features[:n_amplitudes]
        
        # Normalize to unit vector
        norm = qnp.linalg.norm(features)
        if norm > 0:
            features = features / norm
        else:
            # If all zeros, create uniform superposition
            features = qnp.ones(n_amplitudes) / qnp.sqrt(n_amplitudes)
        
        return features
    
    async def embed_legal_concept(
        self,
        classical_features: Union[np.ndarray, torch.Tensor, List[float]],
        concept_type: str = 'query',
        metadata: Optional[Dict[str, Any]] = None
    ) -> QuantumEmbeddingResult:
        """
        Embed a legal concept into quantum state.
        
        Args:
            classical_features: Classical feature vector
            concept_type: Type of legal concept
            metadata: Additional metadata
            
        Returns:
            QuantumEmbeddingResult with quantum state and metrics
        """
        # Convert to numpy array
        if isinstance(classical_features, torch.Tensor):
            classical_features = classical_features.detach().numpy()
        elif isinstance(classical_features, list):
            classical_features = np.array(classical_features)
        
        # Get concept type ID
        concept_type_id = self.concept_types.get(concept_type, 0)
        
        # Set device for the circuit
        circuit = qml.QNode(
            self._quantum_embedding_circuit,
            device=self.device,
            interface="numpy"
        )
        
        # Execute quantum circuit
        quantum_state = circuit(
            classical_features,
            concept_type_id,
            self.embedding_params,
            self.entangling_params,
            self.concept_params
        )
        
        # Extract amplitudes and phases
        amplitudes = qnp.abs(quantum_state)
        phases = qnp.angle(quantum_state)
        
        # Calculate entanglement measure (Meyer-Wallach measure)
        entanglement = self._calculate_entanglement(quantum_state)
        
        # Calculate fidelity with ideal state
        fidelity = self._calculate_fidelity(quantum_state, classical_features)
        
        result = QuantumEmbeddingResult(
            quantum_state=quantum_state,
            amplitudes=amplitudes,
            phases=phases,
            entanglement_measure=entanglement,
            fidelity=fidelity,
            concept_type=concept_type,
            metadata=metadata or {}
        )
        
        logger.debug(f"Embedded {concept_type} concept with entanglement: {entanglement:.3f}")
        return result
    
    def _calculate_entanglement(self, quantum_state: qnp.ndarray) -> float:
        """Calculate Meyer-Wallach entanglement measure."""
        n_qubits = self.n_qubits
        
        # Reshape state for partial traces
        state_tensor = quantum_state.reshape([2] * n_qubits)
        
        entanglement_sum = 0.0
        
        for i in range(n_qubits):
            # Trace out all qubits except i
            axes_to_trace = list(range(n_qubits))
            axes_to_trace.remove(i)
            
            # Calculate reduced density matrix for qubit i
            reduced_state = qnp.tensordot(
                state_tensor,
                qnp.conj(state_tensor),
                axes=(axes_to_trace, axes_to_trace)
            )
            
            # Calculate purity
            purity = qnp.real(qnp.trace(qnp.matmul(reduced_state, reduced_state)))
            entanglement_sum += (1 - purity)
        
        # Normalize by number of qubits
        return entanglement_sum / n_qubits
    
    def _calculate_fidelity(
        self,
        quantum_state: qnp.ndarray,
        classical_features: qnp.ndarray
    ) -> float:
        """Calculate fidelity between quantum state and classical features."""
        # Create ideal quantum state from classical features
        ideal_amplitudes = self._normalize_to_amplitudes(classical_features)
        
        # Pad to full state vector size
        full_ideal_state = qnp.zeros(2 ** self.n_qubits, dtype=complex)
        full_ideal_state[:len(ideal_amplitudes)] = ideal_amplitudes
        
        # Calculate fidelity
        fidelity = qnp.abs(qnp.vdot(quantum_state, full_ideal_state)) ** 2
        return float(fidelity)
    
    async def embed_legal_document(
        self,
        document_features: Dict[str, np.ndarray],
        document_type: str = 'case'
    ) -> Dict[str, QuantumEmbeddingResult]:
        """
        Embed multiple components of a legal document.
        
        Args:
            document_features: Dictionary of feature vectors for different components
            document_type: Type of legal document
            
        Returns:
            Dictionary of quantum embeddings for each component
        """
        embeddings = {}
        
        for component_name, features in document_features.items():
            # Determine concept type based on component
            if 'fact' in component_name.lower():
                concept_type = 'case_fact'
            elif 'precedent' in component_name.lower():
                concept_type = 'precedent'
            elif 'statute' in component_name.lower():
                concept_type = 'statute'
            else:
                concept_type = 'query'
            
            embedding = await self.embed_legal_concept(
                features,
                concept_type=concept_type,
                metadata={'document_type': document_type, 'component': component_name}
            )
            
            embeddings[component_name] = embedding
        
        return embeddings
    
    def compute_quantum_similarity(
        self,
        embedding1: QuantumEmbeddingResult,
        embedding2: QuantumEmbeddingResult
    ) -> float:
        """
        Compute quantum similarity between two embeddings.
        
        Args:
            embedding1: First quantum embedding
            embedding2: Second quantum embedding
            
        Returns:
            Quantum similarity score (0-1)
        """
        # Quantum fidelity as similarity measure
        fidelity = qnp.abs(qnp.vdot(
            embedding1.quantum_state,
            embedding2.quantum_state
        )) ** 2
        
        return float(fidelity)
    
    def update_parameters(self, gradients: Dict[str, qnp.ndarray]) -> None:
        """Update quantum circuit parameters based on gradients."""
        learning_rate = 0.01
        
        if 'embedding_params' in gradients:
            self.embedding_params -= learning_rate * gradients['embedding_params']
        
        if 'entangling_params' in gradients:
            self.entangling_params -= learning_rate * gradients['entangling_params']
        
        if 'concept_params' in gradients:
            self.concept_params -= learning_rate * gradients['concept_params']
    
    def get_parameters(self) -> Dict[str, qnp.ndarray]:
        """Get current quantum circuit parameters."""
        return {
            'embedding_params': self.embedding_params,
            'entangling_params': self.entangling_params,
            'concept_params': self.concept_params
        }
    
    def set_parameters(self, parameters: Dict[str, qnp.ndarray]) -> None:
        """Set quantum circuit parameters."""
        if 'embedding_params' in parameters:
            self.embedding_params = parameters['embedding_params']
        
        if 'entangling_params' in parameters:
            self.entangling_params = parameters['entangling_params']
        
        if 'concept_params' in parameters:
            self.concept_params = parameters['concept_params']
    
    def visualize_embedding(
        self,
        embedding_result: QuantumEmbeddingResult,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Visualize quantum embedding for analysis.
        
        Args:
            embedding_result: Quantum embedding to visualize
            save_path: Optional path to save visualization
            
        Returns:
            Visualization data
        """
        import matplotlib.pyplot as plt
        
        # Create visualization data
        viz_data = {
            'amplitudes': embedding_result.amplitudes,
            'phases': embedding_result.phases,
            'entanglement': embedding_result.entanglement_measure,
            'fidelity': embedding_result.fidelity,
            'concept_type': embedding_result.concept_type
        }
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Amplitude distribution
        ax1.bar(range(len(embedding_result.amplitudes)), embedding_result.amplitudes)
        ax1.set_title('Quantum State Amplitudes')
        ax1.set_xlabel('Basis State')
        ax1.set_ylabel('Amplitude')
        
        # Phase distribution
        ax2.bar(range(len(embedding_result.phases)), embedding_result.phases)
        ax2.set_title('Quantum State Phases')
        ax2.set_xlabel('Basis State')
        ax2.set_ylabel('Phase (radians)')
        
        # Entanglement visualization
        ax3.text(0.5, 0.5, f'Entanglement: {embedding_result.entanglement_measure:.3f}',
                ha='center', va='center', fontsize=16, transform=ax3.transAxes)
        ax3.set_title('Entanglement Measure')
        ax3.axis('off')
        
        # Fidelity visualization
        ax4.text(0.5, 0.5, f'Fidelity: {embedding_result.fidelity:.3f}',
                ha='center', va='center', fontsize=16, transform=ax4.transAxes)
        ax4.set_title('Embedding Fidelity')
        ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.close()
        
        return viz_data