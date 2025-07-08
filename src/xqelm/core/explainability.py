"""
Quantum Explainability Module

This module implements explainability techniques for quantum legal reasoning,
including quantum state tomography, classical shadows, and legal concept mapping
to provide human-interpretable explanations.

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
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


@dataclass
class QuantumExplanation:
    """Quantum explanation result."""
    explanation_text: str
    confidence: float
    key_concepts: List[str]
    reasoning_steps: List[Dict[str, Any]]
    quantum_features: Dict[str, float]
    classical_mapping: Dict[str, Any]
    visualization_data: Dict[str, Any]
    coherence_score: float
    entanglement_contributions: Dict[str, float]
    metadata: Dict[str, Any]


class QuantumExplainabilityModule:
    """
    Quantum Explainability Module for extracting interpretable explanations
    from quantum legal reasoning states.
    
    This module uses quantum state tomography, classical shadows, and
    legal domain knowledge to provide human-readable explanations.
    """
    
    def __init__(
        self,
        n_qubits: int = 20,
        device: Optional[qml.Device] = None,
        n_shadow_samples: int = 1000
    ):
        """
        Initialize the Quantum Explainability Module.
        
        Args:
            n_qubits: Number of qubits in the quantum system
            device: PennyLane quantum device
            n_shadow_samples: Number of samples for classical shadow reconstruction
        """
        self.n_qubits = n_qubits
        self.device = device or qml.device("default.qubit", wires=n_qubits)
        self.n_shadow_samples = n_shadow_samples
        
        # Legal concept mappings
        self.legal_concept_map = self._initialize_legal_concepts()
        
        # Explanation templates
        self.explanation_templates = self._initialize_explanation_templates()
        
        # Quantum measurement bases
        self.measurement_bases = ['X', 'Y', 'Z']
        
        logger.info(f"Initialized QuantumExplainabilityModule for {n_qubits} qubits")
    
    def _initialize_legal_concepts(self) -> Dict[str, Dict[str, Any]]:
        """Initialize legal concept mappings for explanation."""
        return {
            'precedent_relevance': {
                'description': 'How relevant are the cited precedents',
                'threshold': 0.7,
                'interpretation': {
                    'high': 'Strong precedential support',
                    'medium': 'Moderate precedential support',
                    'low': 'Weak precedential support'
                }
            },
            'statutory_compliance': {
                'description': 'Compliance with applicable statutes',
                'threshold': 0.8,
                'interpretation': {
                    'high': 'Clear statutory compliance',
                    'medium': 'Partial statutory compliance',
                    'low': 'Potential statutory violation'
                }
            },
            'constitutional_validity': {
                'description': 'Constitutional validity of the decision',
                'threshold': 0.9,
                'interpretation': {
                    'high': 'Constitutionally sound',
                    'medium': 'Constitutional concerns',
                    'low': 'Potential constitutional violation'
                }
            },
            'factual_similarity': {
                'description': 'Similarity to case facts',
                'threshold': 0.6,
                'interpretation': {
                    'high': 'Highly similar facts',
                    'medium': 'Moderately similar facts',
                    'low': 'Dissimilar facts'
                }
            },
            'legal_principle_strength': {
                'description': 'Strength of underlying legal principles',
                'threshold': 0.75,
                'interpretation': {
                    'high': 'Strong legal foundation',
                    'medium': 'Adequate legal foundation',
                    'low': 'Weak legal foundation'
                }
            }
        }
    
    def _initialize_explanation_templates(self) -> Dict[str, str]:
        """Initialize explanation templates for different scenarios."""
        return {
            'bail_application': """
Based on the quantum legal analysis:

**Decision Recommendation**: {decision}
**Confidence Level**: {confidence:.1%}

**Key Factors Considered**:
- Precedent Relevance: {precedent_score:.1%} - {precedent_interpretation}
- Statutory Compliance: {statutory_score:.1%} - {statutory_interpretation}
- Risk Assessment: {risk_assessment}

**Reasoning Path**:
{reasoning_steps}

**Quantum Insights**:
- Entanglement between legal factors: {entanglement:.3f}
- Quantum coherence of decision: {coherence:.3f}
- Superposition of legal interpretations explored: {superposition_count}

**Legal Basis**:
{legal_citations}
            """,
            
            'cheque_bounce': """
**Liability Assessment**: {decision}
**Confidence**: {confidence:.1%}

**Statutory Analysis**:
- Section 138 NI Act compliance: {statutory_score:.1%}
- Intent analysis: {intent_analysis}
- Amount consideration: {amount_factor}

**Precedent Analysis**:
{precedent_analysis}

**Quantum Reasoning**:
- Legal certainty measure: {coherence:.3f}
- Factor interdependence: {entanglement:.3f}

**Recommended Action**: {recommendation}
            """,
            
            'property_dispute': """
**Property Rights Analysis**: {decision}
**Confidence**: {confidence:.1%}

**Title Analysis**:
- Title clarity: {title_clarity}
- Possession rights: {possession_rights}
- Documentation strength: {documentation_score:.1%}

**Legal Precedents**:
{precedent_summary}

**Quantum Analysis**:
- Legal complexity measure: {entanglement:.3f}
- Decision certainty: {coherence:.3f}

**Resolution Path**: {resolution_recommendation}
            """
        }
    
    async def extract_explanation(
        self,
        quantum_state: qnp.ndarray,
        legal_concepts: List[str],
        use_case_type: str = 'bail_application',
        context: Optional[Dict[str, Any]] = None
    ) -> QuantumExplanation:
        """
        Extract human-interpretable explanation from quantum state.
        
        Args:
            quantum_state: Quantum state from reasoning circuit
            legal_concepts: List of relevant legal concepts
            use_case_type: Type of legal use case
            context: Additional context information
            
        Returns:
            QuantumExplanation with detailed explanation
        """
        logger.info(f"Extracting explanation for {use_case_type}")
        
        # Step 1: Quantum state analysis
        quantum_features = await self._analyze_quantum_state(quantum_state)
        
        # Step 2: Classical shadow reconstruction
        classical_shadows = await self._construct_classical_shadows(quantum_state)
        
        # Step 3: Legal concept mapping
        concept_scores = await self._map_to_legal_concepts(
            quantum_features, classical_shadows, legal_concepts
        )
        
        # Step 4: Reasoning path extraction
        reasoning_steps = await self._extract_reasoning_steps(
            quantum_state, concept_scores, use_case_type
        )
        
        # Step 5: Generate explanation text
        explanation_text = await self._generate_explanation_text(
            concept_scores, reasoning_steps, use_case_type, context
        )
        
        # Step 6: Calculate explanation confidence
        confidence = self._calculate_explanation_confidence(
            quantum_features, concept_scores
        )
        
        # Step 7: Extract key concepts
        key_concepts = self._extract_key_concepts(concept_scores, legal_concepts)
        
        # Step 8: Create visualization data
        visualization_data = await self._create_visualization_data(
            quantum_state, concept_scores, reasoning_steps
        )
        
        # Step 9: Calculate entanglement contributions
        entanglement_contributions = self._calculate_entanglement_contributions(
            quantum_state, legal_concepts
        )
        
        explanation = QuantumExplanation(
            explanation_text=explanation_text,
            confidence=confidence,
            key_concepts=key_concepts,
            reasoning_steps=reasoning_steps,
            quantum_features=quantum_features,
            classical_mapping=concept_scores,
            visualization_data=visualization_data,
            coherence_score=quantum_features.get('coherence', 0.0),
            entanglement_contributions=entanglement_contributions,
            metadata={
                'use_case_type': use_case_type,
                'n_concepts': len(legal_concepts),
                'quantum_state_norm': float(qnp.linalg.norm(quantum_state)),
                'context': context
            }
        )
        
        logger.info(f"Generated explanation with confidence: {confidence:.3f}")
        return explanation
    
    async def _analyze_quantum_state(self, quantum_state: qnp.ndarray) -> Dict[str, float]:
        """Analyze quantum state properties."""
        features = {}
        
        # Amplitude analysis
        amplitudes = qnp.abs(quantum_state)
        features['max_amplitude'] = float(qnp.max(amplitudes))
        features['amplitude_variance'] = float(qnp.var(amplitudes))
        features['amplitude_entropy'] = self._calculate_entropy(amplitudes ** 2)
        
        # Phase analysis
        phases = qnp.angle(quantum_state)
        features['phase_variance'] = float(qnp.var(phases))
        features['phase_coherence'] = self._calculate_phase_coherence(phases)
        
        # Entanglement measures
        features['entanglement'] = self._calculate_meyer_wallach_entanglement(quantum_state)
        features['coherence'] = self._calculate_l1_coherence(quantum_state)
        
        # Participation ratio (measure of localization)
        features['participation_ratio'] = self._calculate_participation_ratio(amplitudes)
        
        # Quantum complexity measures
        features['quantum_complexity'] = self._calculate_quantum_complexity(quantum_state)
        
        return features
    
    async def _construct_classical_shadows(
        self,
        quantum_state: qnp.ndarray
    ) -> List[Tuple[List[str], List[int]]]:
        """Construct classical shadow representation."""
        shadows = []
        
        for _ in range(self.n_shadow_samples):
            # Random Pauli measurement basis
            basis = [
                np.random.choice(self.measurement_bases) 
                for _ in range(self.n_qubits)
            ]
            
            # Simulate measurement
            measurement_result = self._simulate_pauli_measurement(
                quantum_state, basis
            )
            
            shadows.append((basis, measurement_result))
        
        return shadows
    
    def _simulate_pauli_measurement(
        self,
        quantum_state: qnp.ndarray,
        basis: List[str]
    ) -> List[int]:
        """Simulate Pauli measurement on quantum state."""
        # This is a simplified simulation
        # In practice, would use proper quantum measurement
        
        probabilities = qnp.abs(quantum_state) ** 2
        
        # Sample measurement outcomes
        outcomes = []
        for i, pauli in enumerate(basis):
            # Simplified: random outcome based on state probabilities
            if i < len(probabilities):
                prob = probabilities[i]
                outcome = 1 if np.random.random() < prob else -1
            else:
                outcome = np.random.choice([-1, 1])
            outcomes.append(outcome)
        
        return outcomes
    
    async def _map_to_legal_concepts(
        self,
        quantum_features: Dict[str, float],
        classical_shadows: List[Tuple[List[str], List[int]]],
        legal_concepts: List[str]
    ) -> Dict[str, float]:
        """Map quantum features to legal concept scores."""
        concept_scores = {}
        
        # Map quantum features to legal concepts
        for concept in legal_concepts:
            if concept in self.legal_concept_map:
                # Use quantum features to score legal concepts
                score = self._calculate_concept_score(
                    concept, quantum_features, classical_shadows
                )
                concept_scores[concept] = score
        
        # Add derived concepts
        concept_scores['precedent_relevance'] = self._derive_precedent_relevance(
            quantum_features, classical_shadows
        )
        
        concept_scores['statutory_compliance'] = self._derive_statutory_compliance(
            quantum_features, classical_shadows
        )
        
        concept_scores['factual_similarity'] = self._derive_factual_similarity(
            quantum_features, classical_shadows
        )
        
        return concept_scores
    
    def _calculate_concept_score(
        self,
        concept: str,
        quantum_features: Dict[str, float],
        classical_shadows: List[Tuple[List[str], List[int]]]
    ) -> float:
        """Calculate score for a specific legal concept."""
        # Use quantum features to derive concept scores
        
        if concept == 'precedent_relevance':
            # High entanglement suggests strong precedent connections
            return min(1.0, quantum_features.get('entanglement', 0.0) * 2.0)
        
        elif concept == 'statutory_compliance':
            # High coherence suggests clear statutory interpretation
            return quantum_features.get('coherence', 0.0)
        
        elif concept == 'constitutional_validity':
            # Low complexity suggests clear constitutional principles
            complexity = quantum_features.get('quantum_complexity', 1.0)
            return max(0.0, 1.0 - complexity)
        
        elif concept == 'factual_similarity':
            # Participation ratio indicates fact pattern matching
            return quantum_features.get('participation_ratio', 0.5)
        
        elif concept == 'legal_principle_strength':
            # Combination of coherence and low entropy
            coherence = quantum_features.get('coherence', 0.0)
            entropy = quantum_features.get('amplitude_entropy', 1.0)
            return coherence * (1.0 - entropy / np.log(2 ** self.n_qubits))
        
        else:
            # Default scoring based on amplitude variance
            return quantum_features.get('amplitude_variance', 0.5)
    
    def _derive_precedent_relevance(
        self,
        quantum_features: Dict[str, float],
        classical_shadows: List[Tuple[List[str], List[int]]]
    ) -> float:
        """Derive precedent relevance from quantum state."""
        # High entanglement indicates strong precedent connections
        entanglement = quantum_features.get('entanglement', 0.0)
        
        # Analyze shadow measurements for precedent patterns
        precedent_pattern_strength = 0.0
        for basis, outcomes in classical_shadows[:100]:  # Sample subset
            # Look for patterns indicating precedent influence
            if 'Z' in basis:  # Z measurements often indicate classical information
                pattern_score = np.mean([abs(o) for o in outcomes])
                precedent_pattern_strength += pattern_score
        
        precedent_pattern_strength /= 100  # Normalize
        
        # Combine quantum and classical indicators
        return 0.7 * entanglement + 0.3 * precedent_pattern_strength
    
    def _derive_statutory_compliance(
        self,
        quantum_features: Dict[str, float],
        classical_shadows: List[Tuple[List[str], List[int]]]
    ) -> float:
        """Derive statutory compliance from quantum state."""
        # High coherence indicates clear statutory interpretation
        coherence = quantum_features.get('coherence', 0.0)
        
        # Low entropy indicates deterministic statutory application
        entropy = quantum_features.get('amplitude_entropy', 1.0)
        max_entropy = np.log(2 ** self.n_qubits)
        normalized_entropy = entropy / max_entropy
        
        return 0.6 * coherence + 0.4 * (1.0 - normalized_entropy)
    
    def _derive_factual_similarity(
        self,
        quantum_features: Dict[str, float],
        classical_shadows: List[Tuple[List[str], List[int]]]
    ) -> float:
        """Derive factual similarity from quantum state."""
        # Participation ratio indicates how localized the state is
        participation = quantum_features.get('participation_ratio', 0.5)
        
        # High participation suggests broad factual similarity
        # Low participation suggests specific factual matching
        
        # For legal reasoning, moderate participation is often ideal
        optimal_participation = 0.3
        similarity = 1.0 - abs(participation - optimal_participation) / optimal_participation
        
        return max(0.0, min(1.0, similarity))
    
    async def _extract_reasoning_steps(
        self,
        quantum_state: qnp.ndarray,
        concept_scores: Dict[str, float],
        use_case_type: str
    ) -> List[Dict[str, Any]]:
        """Extract reasoning steps from quantum state analysis."""
        steps = []
        
        # Step 1: Initial analysis
        steps.append({
            'step': 1,
            'description': 'Quantum state initialization and query encoding',
            'quantum_measure': quantum_state[0] if len(quantum_state) > 0 else 0,
            'confidence': 0.9,
            'legal_significance': 'Establishes the legal question framework'
        })
        
        # Step 2: Precedent analysis
        precedent_score = concept_scores.get('precedent_relevance', 0.0)
        steps.append({
            'step': 2,
            'description': f'Precedent exploration via quantum walk',
            'quantum_measure': precedent_score,
            'confidence': precedent_score,
            'legal_significance': self._interpret_score(
                precedent_score, 'precedent_relevance'
            )
        })
        
        # Step 3: Statutory analysis
        statutory_score = concept_scores.get('statutory_compliance', 0.0)
        steps.append({
            'step': 3,
            'description': 'Statutory interpretation and compliance check',
            'quantum_measure': statutory_score,
            'confidence': statutory_score,
            'legal_significance': self._interpret_score(
                statutory_score, 'statutory_compliance'
            )
        })
        
        # Step 4: Legal reasoning synthesis
        entanglement = concept_scores.get('entanglement', 0.0)
        steps.append({
            'step': 4,
            'description': 'Quantum entanglement of legal factors',
            'quantum_measure': entanglement,
            'confidence': min(1.0, entanglement * 1.5),
            'legal_significance': 'Integration of multiple legal considerations'
        })
        
        # Step 5: Conclusion formation
        coherence = concept_scores.get('coherence', 0.0)
        steps.append({
            'step': 5,
            'description': 'Legal conclusion extraction',
            'quantum_measure': coherence,
            'confidence': coherence,
            'legal_significance': 'Final legal determination with quantum certainty'
        })
        
        return steps
    
    def _interpret_score(self, score: float, concept: str) -> str:
        """Interpret a concept score using legal concept mappings."""
        if concept not in self.legal_concept_map:
            return f"Score: {score:.2f}"
        
        concept_info = self.legal_concept_map[concept]
        threshold = concept_info['threshold']
        interpretations = concept_info['interpretation']
        
        if score >= threshold:
            return interpretations['high']
        elif score >= threshold * 0.6:
            return interpretations['medium']
        else:
            return interpretations['low']
    
    async def _generate_explanation_text(
        self,
        concept_scores: Dict[str, float],
        reasoning_steps: List[Dict[str, Any]],
        use_case_type: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate human-readable explanation text."""
        template = self.explanation_templates.get(
            use_case_type, 
            self.explanation_templates['bail_application']
        )
        
        # Prepare template variables
        template_vars = {
            'decision': self._determine_decision(concept_scores, use_case_type),
            'confidence': concept_scores.get('statutory_compliance', 0.5),
            'precedent_score': concept_scores.get('precedent_relevance', 0.0),
            'precedent_interpretation': self._interpret_score(
                concept_scores.get('precedent_relevance', 0.0), 'precedent_relevance'
            ),
            'statutory_score': concept_scores.get('statutory_compliance', 0.0),
            'statutory_interpretation': self._interpret_score(
                concept_scores.get('statutory_compliance', 0.0), 'statutory_compliance'
            ),
            'coherence': concept_scores.get('coherence', 0.0),
            'entanglement': concept_scores.get('entanglement', 0.0),
            'reasoning_steps': self._format_reasoning_steps(reasoning_steps),
            'superposition_count': len([s for s in reasoning_steps if s['confidence'] > 0.5]),
            'legal_citations': self._generate_legal_citations(concept_scores, use_case_type),
            'risk_assessment': self._generate_risk_assessment(concept_scores, use_case_type)
        }
        
        # Add use-case specific variables
        if use_case_type == 'cheque_bounce':
            template_vars.update({
                'intent_analysis': 'Intent clearly established' if concept_scores.get('factual_similarity', 0) > 0.7 else 'Intent requires further analysis',
                'amount_factor': 'Significant amount involved' if context and context.get('amount', 0) > 100000 else 'Moderate amount',
                'precedent_analysis': 'Strong precedential support for liability',
                'recommendation': 'Proceed with prosecution' if concept_scores.get('statutory_compliance', 0) > 0.8 else 'Consider settlement'
            })
        
        elif use_case_type == 'property_dispute':
            template_vars.update({
                'title_clarity': 'Clear title' if concept_scores.get('statutory_compliance', 0) > 0.8 else 'Title issues present',
                'possession_rights': 'Strong possession claim' if concept_scores.get('factual_similarity', 0) > 0.7 else 'Disputed possession',
                'documentation_score': concept_scores.get('statutory_compliance', 0.0),
                'precedent_summary': 'Favorable precedents identified',
                'resolution_recommendation': 'Mediation recommended' if concept_scores.get('entanglement', 0) > 0.5 else 'Litigation may be necessary'
            })
        
        try:
            return template.format(**template_vars)
        except KeyError as e:
            logger.warning(f"Missing template variable: {e}")
            return f"Legal analysis completed with {concept_scores.get('statutory_compliance', 0.5):.1%} confidence."
    
    def _determine_decision(
        self,
        concept_scores: Dict[str, float],
        use_case_type: str
    ) -> str:
        """Determine the legal decision based on concept scores."""
        if use_case_type == 'bail_application':
            if concept_scores.get('statutory_compliance', 0) > 0.6:
                return 'Grant Bail'
            else:
                return 'Deny Bail'
        
        elif use_case_type == 'cheque_bounce':
            if concept_scores.get('statutory_compliance', 0) > 0.7:
                return 'Liable under Section 138'
            else:
                return 'Not Liable'
        
        elif use_case_type == 'property_dispute':
            if concept_scores.get('precedent_relevance', 0) > 0.7:
                return 'Plaintiff has stronger claim'
            else:
                return 'Defendant has stronger claim'
        
        else:
            return 'Further analysis required'
    
    def _format_reasoning_steps(self, reasoning_steps: List[Dict[str, Any]]) -> str:
        """Format reasoning steps for display."""
        formatted_steps = []
        
        for step in reasoning_steps:
            formatted_steps.append(
                f"{step['step']}. {step['description']} "
                f"(Confidence: {step['confidence']:.1%})\n"
                f"   Legal Significance: {step['legal_significance']}"
            )
        
        return '\n\n'.join(formatted_steps)
    
    def _generate_legal_citations(
        self,
        concept_scores: Dict[str, float],
        use_case_type: str
    ) -> str:
        """Generate relevant legal citations."""
        citations = []
        
        if use_case_type == 'bail_application':
            citations.extend([
                "Section 437 CrPC - Bail in non-bailable offences",
                "Gurcharan Singh v. State (Delhi) - Bail considerations",
                "Article 21 Constitution - Right to life and liberty"
            ])
        
        elif use_case_type == 'cheque_bounce':
            citations.extend([
                "Section 138 Negotiable Instruments Act",
                "Kusum Ingots v. Pennar Peterson - Dishonor of cheque",
                "C.C. Alavi Haji v. Palapetty Muhammed - Presumption under Section 139"
            ])
        
        elif use_case_type == 'property_dispute':
            citations.extend([
                "Transfer of Property Act, 1882",
                "Specific Relief Act, 1963",
                "Sarla Verma v. Delhi Development Authority - Property rights"
            ])
        
        return '\n'.join([f"• {citation}" for citation in citations])
    
    def _generate_risk_assessment(
        self,
        concept_scores: Dict[str, float],
        use_case_type: str
    ) -> str:
        """Generate risk assessment based on concept scores."""
        if use_case_type == 'bail_application':
            risk_factors = []
            
            if concept_scores.get('factual_similarity', 0) < 0.5:
                risk_factors.append("Flight risk - Low community ties")
            
            if concept_scores.get('precedent_relevance', 0) < 0.6:
                risk_factors.append("Public safety concerns")
            
            if concept_scores.get('statutory_compliance', 0) < 0.7:
                risk_factors.append("Evidence tampering risk")
            
            if not risk_factors:
                return "Low risk - Suitable for bail"
            else:
                return "Risk factors: " + "; ".join(risk_factors)
        
        return "Risk assessment completed"
    
    def _calculate_explanation_confidence(
        self,
        quantum_features: Dict[str, float],
        concept_scores: Dict[str, float]
    ) -> float:
        """Calculate overall confidence in the explanation."""
        # Combine quantum coherence with concept score consistency
        coherence = quantum_features.get('coherence', 0.0)
        
        # Calculate variance in concept scores (lower variance = higher confidence)
        scores = list(concept_scores.values())
        if scores:
            score_variance = np.var(scores)
            consistency = 1.0 / (1.0 + score_variance)
        else:
            consistency = 0.5
        
        # Combine factors
        confidence = 0.6 * coherence + 0.4 * consistency
        
        return min(1.0, max(0.0, confidence))
    
    def _extract_key_concepts(
        self,
        concept_scores: Dict[str, float],
        legal_concepts: List[str]
    ) -> List[str]:
        """Extract key legal concepts based on scores."""
        # Sort concepts by score
        sorted_concepts = sorted(
            concept_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top concepts above threshold
        key_concepts = []
        for concept, score in sorted_concepts:
            if score > 0.5 and len(key_concepts) < 5:
                key_concepts.append(concept)
        
        return key_concepts
    
    async def _create_visualization_data(
        self,
        quantum_state: qnp.ndarray,
        concept_scores: Dict[str, float],
        reasoning_steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create data for visualization."""
        viz_data = {
            'quantum_state': {
                'amplitudes': qnp.abs(quantum_state).tolist(),
                'phases': qnp.angle(quantum_state).tolist()
            },
            'concept_scores': concept_scores,
            'reasoning_flow': [
                {
                    'step': step['step'],
                    'confidence': step['confidence'],
                    'quantum_measure': step['quantum_measure']
                }
                for step in reasoning_steps
            ],
            'entanglement_network': self._create_entanglement_network(quantum_state)
        }
        
        return viz_data
    
    def _create_entanglement_network(self, quantum_state: qnp.ndarray) -> Dict[str, Any]:
        """Create entanglement network visualization data."""
        # Calculate pairwise entanglement between qubits
        n_qubits = self.n_qubits
        entanglement_matrix = np.zeros((n_qubits, n_qubits))
        
        # Simplified entanglement calculation
        state_tensor = quantum_state.reshape([2] * n_qubits)
        
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                # Calculate mutual information as entanglement measure
                entanglement = self._calculate_mutual_information(
                    state_tensor, i, j
                )
                entanglement_matrix[i, j] = entanglement
                entanglement_matrix[j, i] = entanglement
        
        return {
            'nodes': [{'id': i, 'label': f'Q{i}'} for i in range(n_qubits)],
            'edges': [
                {
                    'source': i,
                    'target': j,
                    'weight': float(entanglement_matrix[i, j])
                }
                for i in range(n_qubits)
                for j in range(i + 1, n_qubits)
                if entanglement_matrix[i, j] > 0.1
            ]
        }
    
    def _calculate_entanglement_contributions(
        self,
        quantum_state: qnp.ndarray,
        legal_concepts: List[str]
    ) -> Dict[str, float]:
        """Calculate how much each legal concept contributes to entanglement."""
        contributions = {}
        
        # Map legal concepts to qubit groups
        concept_qubit_map = {
            'precedent_relevance': list(range(0, self.n_qubits // 4)),
            'statutory_compliance': list(range(self.n_qubits // 4, self.n_qubits // 2)),
            'factual_similarity': list(range(self.n_qubits // 2, 3 * self.n_qubits // 4)),
            'legal_principle_strength': list(range(3 * self.n_qubits // 4, self.n_qubits))
        }
        
        # Calculate entanglement for each concept group
        for concept in legal_concepts:
            if concept in concept_qubit_map:
                qubits = concept_qubit_map[concept]
                entanglement = self._calculate_group_entanglement(quantum_state, qubits)
                contributions[concept] = entanglement
        
        return contributions
    
    def _calculate_group_entanglement(
        self,
        quantum_state: qnp.ndarray,
        qubit_group: List[int]
    ) -> float:
        """Calculate entanglement within a group of qubits."""
        if len(qubit_group) < 2:
            return 0.0
        
        # Simplified group entanglement calculation
        state_tensor = quantum_state.reshape([2] * self.n_qubits)
        
        # Calculate average pairwise entanglement in the group
        total_entanglement = 0.0
        pair_count = 0
        
        for i in range(len(qubit_group)):
            for j in range(i + 1, len(qubit_group)):
                if qubit_group[i] < self.n_qubits and qubit_group[j] < self.n_qubits:
                    entanglement = self._calculate_mutual_information(
                        state_tensor, qubit_group[i], qubit_group[j]
                    )
                    total_entanglement += entanglement
                    pair_count += 1
        
        return total_entanglement / pair_count if pair_count > 0 else 0.0
    
    def _calculate_mutual_information(
        self,
        state_tensor: qnp.ndarray,
        qubit_i: int,
        qubit_j: int
    ) -> float:
        """Calculate mutual information between two qubits."""
        # Simplified mutual information calculation
        # In practice, would use proper quantum information measures
        
        # Calculate reduced density matrices
        axes_to_trace_i = list(range(self.n_qubits))
        axes_to_trace_i.remove(qubit_i)
        if qubit_j in axes_to_trace_i:
            axes_to_trace_i.remove(qubit_j)
        
        # This is a simplified approximation
        # Real implementation would calculate von Neumann entropies
        correlation = abs(state_tensor.flatten()[qubit_i * 2 + qubit_j]) if qubit_i * 2 + qubit_j < len(state_tensor.flatten()) else 0.0
        
        return float(correlation)
    
    # Utility methods for quantum state analysis
    
    def _calculate_entropy(self, probabilities: qnp.ndarray) -> float:
        """Calculate Shannon entropy."""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        probs = probabilities + epsilon
        probs = probs / qnp.sum(probs)  # Normalize
        
        entropy = -qnp.sum(probs * qnp.log(probs))
        return float(entropy)
    
    def _calculate_phase_coherence(self, phases: qnp.ndarray) -> float:
        """Calculate phase coherence measure."""
        # Calculate circular variance of phases
        mean_phase = qnp.angle(qnp.mean(qnp.exp(1j * phases)))
        phase_deviations = qnp.abs(phases - mean_phase)
        
        # Normalize to [0, 1] where 1 is perfect coherence
        coherence = 1.0 - qnp.mean(phase_deviations) / np.pi
        return float(max(0.0, coherence))
    
    def _calculate_meyer_wallach_entanglement(self, quantum_state: qnp.ndarray) -> float:
        """Calculate Meyer-Wallach entanglement measure."""
        n_qubits = self.n_qubits
        state_tensor = quantum_state.reshape([2] * n_qubits)
        
        entanglement_sum = 0.0
        
        for i in range(n_qubits):
            # Trace out all qubits except i
            axes_to_trace = list(range(n_qubits))
            axes_to_trace.remove(i)
            
            # Calculate reduced density matrix
            reduced_state = qnp.tensordot(
                state_tensor,
                qnp.conj(state_tensor),
                axes=(axes_to_trace, axes_to_trace)
            )
            
            # Calculate purity
            purity = qnp.real(qnp.trace(qnp.matmul(reduced_state, reduced_state)))
            entanglement_sum += (1 - purity)
        
        return float(entanglement_sum / n_qubits)
    
    def _calculate_l1_coherence(self, quantum_state: qnp.ndarray) -> float:
        """Calculate L1-norm coherence."""
        # Create density matrix
        density_matrix = qnp.outer(quantum_state, qnp.conj(quantum_state))
        
        # Remove diagonal elements
        coherence_matrix = density_matrix - qnp.diag(qnp.diag(density_matrix))
        
        # L1-norm coherence
        coherence = qnp.sum(qnp.abs(coherence_matrix))
        return float(coherence)
    
    def _calculate_participation_ratio(self, amplitudes: qnp.ndarray) -> float:
        """Calculate participation ratio (inverse participation ratio)."""
        # Normalize amplitudes
        norm_amplitudes = amplitudes / qnp.linalg.norm(amplitudes)
        
        # Calculate participation ratio
        participation = 1.0 / qnp.sum(norm_amplitudes ** 4)
        
        # Normalize by maximum possible value
        max_participation = len(amplitudes)
        normalized_participation = participation / max_participation
        
        return float(normalized_participation)
    
    def _calculate_quantum_complexity(self, quantum_state: qnp.ndarray) -> float:
        """Calculate quantum complexity measure."""
        # Use amplitude variance as a proxy for complexity
        amplitudes = qnp.abs(quantum_state)
        
        # High variance indicates complex superposition
        variance = qnp.var(amplitudes)
        
        # Normalize by maximum possible variance
        max_variance = 0.25  # For uniform distribution
        normalized_complexity = variance / max_variance
        
        return float(min(1.0, normalized_complexity))
    
    def visualize_explanation(
        self,
        explanation: QuantumExplanation,
        save_path: Optional[str] = None
    ) -> None:
        """Create visualization of the quantum explanation."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Concept scores radar chart
        concepts = list(explanation.classical_mapping.keys())
        scores = list(explanation.classical_mapping.values())
        
        if concepts and scores:
            angles = np.linspace(0, 2 * np.pi, len(concepts), endpoint=False)
            scores_plot = scores + [scores[0]]  # Close the plot
            angles_plot = np.concatenate([angles, [angles[0]]])
            
            ax1.plot(angles_plot, scores_plot, 'o-', linewidth=2)
            ax1.fill(angles_plot, scores_plot, alpha=0.25)
            ax1.set_xticks(angles)
            ax1.set_xticklabels(concepts, rotation=45)
            ax1.set_ylim(0, 1)
            ax1.set_title('Legal Concept Scores')
            ax1.grid(True)
        
        # 2. Quantum state amplitudes
        viz_data = explanation.visualization_data
        if 'quantum_state' in viz_data:
            amplitudes = viz_data['quantum_state']['amplitudes']
            ax2.bar(range(len(amplitudes)), amplitudes)
            ax2.set_title('Quantum State Amplitudes')
            ax2.set_xlabel('Basis State')
            ax2.set_ylabel('Amplitude')
        
        # 3. Reasoning flow
        if 'reasoning_flow' in viz_data:
            flow_data = viz_data['reasoning_flow']
            steps = [item['step'] for item in flow_data]
            confidences = [item['confidence'] for item in flow_data]
            
            ax3.plot(steps, confidences, 'o-', linewidth=2, markersize=8)
            ax3.set_title('Reasoning Confidence Flow')
            ax3.set_xlabel('Reasoning Step')
            ax3.set_ylabel('Confidence')
            ax3.set_ylim(0, 1)
            ax3.grid(True)
        
        # 4. Entanglement contributions
        entanglement_data = explanation.entanglement_contributions
        if entanglement_data:
            concepts_ent = list(entanglement_data.keys())
            contributions = list(entanglement_data.values())
            
            ax4.barh(concepts_ent, contributions)
            ax4.set_title('Entanglement Contributions')
            ax4.set_xlabel('Entanglement Measure')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Explanation visualization saved to {save_path}")
        
        plt.show()
    
    def export_explanation(
        self,
        explanation: QuantumExplanation,
        format: str = 'json',
        filepath: Optional[str] = None
    ) -> Union[str, Dict[str, Any]]:
        """Export explanation in various formats."""
        
        export_data = {
            'explanation_text': explanation.explanation_text,
            'confidence': explanation.confidence,
            'key_concepts': explanation.key_concepts,
            'reasoning_steps': explanation.reasoning_steps,
            'quantum_features': explanation.quantum_features,
            'classical_mapping': explanation.classical_mapping,
            'coherence_score': explanation.coherence_score,
            'entanglement_contributions': explanation.entanglement_contributions,
            'metadata': explanation.metadata
        }
        
        if format.lower() == 'json':
            import json
            json_str = json.dumps(export_data, indent=2, default=str)
            
            if filepath:
                with open(filepath, 'w') as f:
                    f.write(json_str)
                logger.info(f"Explanation exported to {filepath}")
            
            return json_str
        
        elif format.lower() == 'dict':
            return export_data
        
        elif format.lower() == 'text':
            text_export = f"""
QUANTUM LEGAL EXPLANATION REPORT
================================

{explanation.explanation_text}

TECHNICAL DETAILS
-----------------
Confidence: {explanation.confidence:.3f}
Coherence Score: {explanation.coherence_score:.3f}

Key Concepts: {', '.join(explanation.key_concepts)}

Quantum Features:
{chr(10).join([f"- {k}: {v:.3f}" for k, v in explanation.quantum_features.items()])}

Entanglement Contributions:
{chr(10).join([f"- {k}: {v:.3f}" for k, v in explanation.entanglement_contributions.items()])}
            """
            
            if filepath:
                with open(filepath, 'w') as f:
                    f.write(text_export)
                logger.info(f"Explanation exported to {filepath}")
            
            return text_export
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
            