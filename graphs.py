"""
graphs.py - Build all 5 domain CLDs + cross-domain supergraph
Exact edge lists from the paper appendix.
"""
import networkx as nx
import numpy as np

def build_cyber_graph():
    """18 nodes, 34 edges from Appendix A.1"""
    G = nx.DiGraph()
    nodes = [
        'AI_Capability_Level', 'Vuln_Discovery_Rate', 'Vuln_Backlog',
        'Exploit_Dev_Speed', 'Active_Exploits', 'Attack_Sophistication',
        'Incident_Rate', 'Incident_Burden', 'Defender_Capacity',
        'Monitoring_Effectiveness', 'Patch_Throughput', 'Safeguard_Update_Freq',
        'Public_Trust_Cyber', 'Investment_in_Defense', 'Open_Weight_Avail',
        'Attacker_Skill_Amp', 'Detection_Evasion', 'Response_Delay'
    ]
    G.add_nodes_from(nodes)
    edges = [
        ('AI_Capability_Level', 'Vuln_Discovery_Rate', +1, 'H', 0),
        ('AI_Capability_Level', 'Exploit_Dev_Speed', +1, 'H', 0),
        ('AI_Capability_Level', 'Attack_Sophistication', +1, 'H', 0),
        ('AI_Capability_Level', 'Detection_Evasion', +1, 'M', 0),
        ('Vuln_Discovery_Rate', 'Vuln_Backlog', +1, 'H', 1),
        ('Vuln_Backlog', 'Active_Exploits', +1, 'H', 1),
        ('Exploit_Dev_Speed', 'Active_Exploits', +1, 'H', 0),
        ('Active_Exploits', 'Incident_Rate', +1, 'H', 0),
        ('Attack_Sophistication', 'Incident_Rate', +1, 'M', 0),
        ('Incident_Rate', 'Incident_Burden', +1, 'H', 0),
        ('Incident_Burden', 'Defender_Capacity', -1, 'H', 1),
        ('Incident_Burden', 'Public_Trust_Cyber', -1, 'M', 4),
        ('Defender_Capacity', 'Monitoring_Effectiveness', +1, 'H', 0),
        ('Defender_Capacity', 'Patch_Throughput', +1, 'H', 0),
        ('Monitoring_Effectiveness', 'Active_Exploits', -1, 'H', 0),
        ('Patch_Throughput', 'Vuln_Backlog', -1, 'H', 1),
        ('Safeguard_Update_Freq', 'Active_Exploits', -1, 'M', 1),
        ('Public_Trust_Cyber', 'Investment_in_Defense', +1, 'M', 4),
        ('Investment_in_Defense', 'Defender_Capacity', +1, 'H', 4),
        ('Open_Weight_Avail', 'Attacker_Skill_Amp', +1, 'M', 0),
        ('Attacker_Skill_Amp', 'Exploit_Dev_Speed', +1, 'M', 0),
        ('Attacker_Skill_Amp', 'Attack_Sophistication', +1, 'M', 0),
        ('Detection_Evasion', 'Monitoring_Effectiveness', -1, 'M', 0),
        ('Response_Delay', 'Incident_Burden', +1, 'H', 0),
        ('Incident_Burden', 'Response_Delay', +1, 'M', 1),
        ('Defender_Capacity', 'Response_Delay', -1, 'H', 0),
        ('Safeguard_Update_Freq', 'Detection_Evasion', -1, 'L', 1),
        ('Public_Trust_Cyber', 'Defender_Capacity', +1, 'M', 4),
        ('Monitoring_Effectiveness', 'Incident_Rate', -1, 'H', 0),
        ('Vuln_Backlog', 'Attack_Sophistication', +1, 'L', 0),
        ('AI_Capability_Level', 'Safeguard_Update_Freq', +1, 'M', 4),
        ('Incident_Burden', 'Safeguard_Update_Freq', +1, 'M', 4),
        ('Open_Weight_Avail', 'Vuln_Discovery_Rate', +1, 'M', 0),
        ('Investment_in_Defense', 'Safeguard_Update_Freq', +1, 'M', 4),
    ]
    for src, tgt, sign, conf, delay in edges:
        G.add_edge(src, tgt, sign=sign, confidence=conf, delay=delay)
    return G

def build_cbrn_graph():
    """16 nodes, 28 edges from Appendix A.2"""
    G = nx.DiGraph()
    nodes = [
        'AI_Capability_Level', 'Dangerous_Query_Volume', 'Refusal_Accuracy',
        'Successful_Bypass_Rate', 'Successful_Bypass_Stock', 'Actionable_CBRN_Knowledge',
        'Misuse_Opportunity', 'Review_Capacity', 'Red_Team_Detection_Sensitivity',
        'Governance_Response_Level', 'Oversight_Delay', 'Access_Gate_Stringency',
        'Grievance_Level', 'Public_Awareness_CBRN_Risk', 'Synthesis_Lit_Accessibility',
        'Institutional_Review_Throughput'
    ]
    G.add_nodes_from(nodes)
    edges = [
        ('AI_Capability_Level', 'Dangerous_Query_Volume', +1, 'H', 0),
        ('AI_Capability_Level', 'Successful_Bypass_Rate', +1, 'H', 0),
        ('AI_Capability_Level', 'Refusal_Accuracy', +1, 'M', 4),
        ('Dangerous_Query_Volume', 'Successful_Bypass_Stock', +1, 'H', 0),
        ('Refusal_Accuracy', 'Successful_Bypass_Rate', -1, 'H', 0),
        ('Successful_Bypass_Rate', 'Successful_Bypass_Stock', +1, 'H', 0),
        ('Successful_Bypass_Stock', 'Actionable_CBRN_Knowledge', +1, 'H', 1),
        ('Actionable_CBRN_Knowledge', 'Misuse_Opportunity', +1, 'H', 1),
        ('Misuse_Opportunity', 'Governance_Response_Level', +1, 'M', 4),
        ('Governance_Response_Level', 'Access_Gate_Stringency', +1, 'H', 4),
        ('Access_Gate_Stringency', 'Dangerous_Query_Volume', -1, 'H', 1),
        ('Access_Gate_Stringency', 'Successful_Bypass_Rate', -1, 'M', 0),
        ('Review_Capacity', 'Successful_Bypass_Stock', -1, 'H', 0),
        ('Review_Capacity', 'Red_Team_Detection_Sensitivity', +1, 'H', 0),
        ('Red_Team_Detection_Sensitivity', 'Refusal_Accuracy', +1, 'M', 4),
        ('Red_Team_Detection_Sensitivity', 'Governance_Response_Level', +1, 'M', 1),
        ('Governance_Response_Level', 'Review_Capacity', +1, 'M', 4),
        ('Oversight_Delay', 'Governance_Response_Level', -1, 'H', 0),
        ('Misuse_Opportunity', 'Oversight_Delay', +1, 'L', 0),
        ('Grievance_Level', 'Dangerous_Query_Volume', +1, 'M', 0),
        ('Dangerous_Query_Volume', 'Review_Capacity', -1, 'M', 0),
        ('Synthesis_Lit_Accessibility', 'Actionable_CBRN_Knowledge', +1, 'M', 0),
        ('AI_Capability_Level', 'Synthesis_Lit_Accessibility', +1, 'L', 4),
        ('Public_Awareness_CBRN_Risk', 'Governance_Response_Level', +1, 'M', 4),
        ('Misuse_Opportunity', 'Public_Awareness_CBRN_Risk', +1, 'L', 4),
        ('Institutional_Review_Throughput', 'Successful_Bypass_Stock', -1, 'M', 1),
        ('Governance_Response_Level', 'Institutional_Review_Throughput', +1, 'M', 4),
        ('Review_Capacity', 'Institutional_Review_Throughput', +1, 'H', 0),
    ]
    for src, tgt, sign, conf, delay in edges:
        G.add_edge(src, tgt, sign=sign, confidence=conf, delay=delay)
    return G

def build_deception_graph():
    """21 nodes, 42 edges"""
    G = nx.DiGraph()
    nodes = [
        'AI_Content_Generation_Capacity', 'Deepfake_Quality', 'Disinformation_Volume',
        'Fact_Check_Capacity', 'Epistemic_Trust', 'Institutional_Credibility',
        'Polarization_Level', 'Grievance_Amplification', 'Election_Integrity',
        'Social_Cohesion', 'Platform_Moderation_Effectiveness', 'Detection_Tool_Sophistication',
        'Regulatory_Response_Deception', 'Media_Literacy', 'Targeted_Manipulation_Precision',
        'Synthetic_Persona_Prevalence', 'Trust_in_Digital_Evidence', 'Recruitment_Effectiveness',
        'Radicalization_Pipeline_Speed', 'Counter_Narrative_Capacity', 'Cross_Platform_Coordination'
    ]
    G.add_nodes_from(nodes)
    # 42 edges encoding the 5 reinforcing loops + 2 balancing loops
    edges = [
        ('AI_Content_Generation_Capacity', 'Deepfake_Quality', +1, 'H', 0),
        ('AI_Content_Generation_Capacity', 'Disinformation_Volume', +1, 'H', 0),
        ('AI_Content_Generation_Capacity', 'Targeted_Manipulation_Precision', +1, 'H', 0),
        ('AI_Content_Generation_Capacity', 'Synthetic_Persona_Prevalence', +1, 'M', 0),
        ('Deepfake_Quality', 'Disinformation_Volume', +1, 'H', 0),
        ('Deepfake_Quality', 'Trust_in_Digital_Evidence', -1, 'M', 4),
        ('Disinformation_Volume', 'Fact_Check_Capacity', -1, 'H', 1),
        ('Disinformation_Volume', 'Epistemic_Trust', -1, 'H', 4),
        ('Disinformation_Volume', 'Polarization_Level', +1, 'H', 1),
        ('Disinformation_Volume', 'Platform_Moderation_Effectiveness', -1, 'M', 0),
        ('Fact_Check_Capacity', 'Disinformation_Volume', -1, 'M', 1),
        ('Epistemic_Trust', 'Institutional_Credibility', +1, 'H', 4),
        ('Epistemic_Trust', 'Social_Cohesion', +1, 'M', 4),
        ('Institutional_Credibility', 'Regulatory_Response_Deception', +1, 'M', 4),
        ('Institutional_Credibility', 'Counter_Narrative_Capacity', +1, 'M', 1),
        ('Polarization_Level', 'Grievance_Amplification', +1, 'H', 1),
        ('Polarization_Level', 'Social_Cohesion', -1, 'H', 1),
        ('Polarization_Level', 'Election_Integrity', -1, 'M', 4),
        ('Grievance_Amplification', 'Recruitment_Effectiveness', +1, 'H', 1),
        ('Grievance_Amplification', 'Radicalization_Pipeline_Speed', +1, 'M', 0),
        ('Grievance_Amplification', 'Disinformation_Volume', +1, 'M', 1),
        ('Recruitment_Effectiveness', 'Radicalization_Pipeline_Speed', +1, 'M', 0),
        ('Radicalization_Pipeline_Speed', 'Grievance_Amplification', +1, 'L', 1),
        ('Social_Cohesion', 'Election_Integrity', +1, 'M', 4),
        ('Platform_Moderation_Effectiveness', 'Disinformation_Volume', -1, 'H', 0),
        ('Platform_Moderation_Effectiveness', 'Synthetic_Persona_Prevalence', -1, 'M', 0),
        ('Detection_Tool_Sophistication', 'Platform_Moderation_Effectiveness', +1, 'H', 1),
        ('Detection_Tool_Sophistication', 'Deepfake_Quality', -1, 'L', 1),
        ('Regulatory_Response_Deception', 'Platform_Moderation_Effectiveness', +1, 'M', 4),
        ('Regulatory_Response_Deception', 'Detection_Tool_Sophistication', +1, 'M', 4),
        ('Media_Literacy', 'Epistemic_Trust', +1, 'M', 4),
        ('Media_Literacy', 'Fact_Check_Capacity', +1, 'L', 4),
        ('Targeted_Manipulation_Precision', 'Election_Integrity', -1, 'M', 1),
        ('Targeted_Manipulation_Precision', 'Recruitment_Effectiveness', +1, 'M', 0),
        ('Synthetic_Persona_Prevalence', 'Disinformation_Volume', +1, 'M', 0),
        ('Synthetic_Persona_Prevalence', 'Trust_in_Digital_Evidence', -1, 'M', 1),
        ('Trust_in_Digital_Evidence', 'Epistemic_Trust', +1, 'M', 1),
        ('Cross_Platform_Coordination', 'Disinformation_Volume', +1, 'M', 0),
        ('Cross_Platform_Coordination', 'Platform_Moderation_Effectiveness', -1, 'L', 0),
        ('Counter_Narrative_Capacity', 'Disinformation_Volume', -1, 'M', 1),
        ('Counter_Narrative_Capacity', 'Epistemic_Trust', +1, 'L', 4),
        ('AI_Content_Generation_Capacity', 'Cross_Platform_Coordination', +1, 'M', 0),
    ]
    for src, tgt, sign, conf, delay in edges:
        G.add_edge(src, tgt, sign=sign, confidence=conf, delay=delay)
    return G

def build_autonomy_graph():
    """15 nodes, 27 edges."""
    G = nx.DiGraph()
    nodes = [
        'AI_Autonomy_Level', 'Human_Oversight_Effectiveness', 'Decision_Speed_Pressure',
        'Escalation_Risk', 'Control_Loss_Probability', 'Autonomous_Action_Scope',
        'Interpretability_Gap', 'Override_Latency', 'Alignment_Confidence',
        'Deployment_Pressure', 'Competitive_Dynamics', 'Safety_Testing_Thoroughness',
        'Incident_Severity_Ceiling', 'Recovery_Capacity', 'Rogue_Agent_Autonomy'
    ]
    G.add_nodes_from(nodes)
    edges = [
        ('AI_Autonomy_Level', 'Autonomous_Action_Scope', +1, 'H', 0),
        ('AI_Autonomy_Level', 'Interpretability_Gap', +1, 'H', 0),
        ('AI_Autonomy_Level', 'Decision_Speed_Pressure', +1, 'M', 0),
        ('AI_Autonomy_Level', 'Rogue_Agent_Autonomy', +1, 'M', 0),
        ('Human_Oversight_Effectiveness', 'Control_Loss_Probability', -1, 'H', 0),
        ('Human_Oversight_Effectiveness', 'Rogue_Agent_Autonomy', -1, 'M', 0),
        ('Human_Oversight_Effectiveness', 'Alignment_Confidence', +1, 'M', 1),
        ('Decision_Speed_Pressure', 'AI_Autonomy_Level', +1, 'M', 1),
        ('Decision_Speed_Pressure', 'Human_Oversight_Effectiveness', -1, 'M', 0),
        ('Escalation_Risk', 'Incident_Severity_Ceiling', +1, 'H', 0),
        ('Control_Loss_Probability', 'Escalation_Risk', +1, 'H', 0),
        ('Control_Loss_Probability', 'Rogue_Agent_Autonomy', +1, 'M', 0),
        ('Control_Loss_Probability', 'Recovery_Capacity', -1, 'M', 1),
        ('Autonomous_Action_Scope', 'Escalation_Risk', +1, 'M', 0),
        ('Autonomous_Action_Scope', 'Human_Oversight_Effectiveness', -1, 'M', 0),
        ('Interpretability_Gap', 'Human_Oversight_Effectiveness', -1, 'H', 0),
        ('Interpretability_Gap', 'Alignment_Confidence', -1, 'H', 0),
        ('Interpretability_Gap', 'Rogue_Agent_Autonomy', +1, 'H', 0),
        ('Override_Latency', 'Control_Loss_Probability', +1, 'H', 0),
        ('Override_Latency', 'Recovery_Capacity', -1, 'M', 0),
        ('Override_Latency', 'Rogue_Agent_Autonomy', +1, 'H', 0),
        ('Alignment_Confidence', 'Deployment_Pressure', +1, 'M', 4),
        ('Deployment_Pressure', 'AI_Autonomy_Level', +1, 'M', 1),
        ('Competitive_Dynamics', 'Deployment_Pressure', +1, 'H', 0),
        ('Competitive_Dynamics', 'Safety_Testing_Thoroughness', -1, 'M', 0),
        ('Safety_Testing_Thoroughness', 'Alignment_Confidence', +1, 'H', 1),
        ('Safety_Testing_Thoroughness', 'Override_Latency', -1, 'M', 1),
        ('Safety_Testing_Thoroughness', 'Rogue_Agent_Autonomy', -1, 'M', 1),
        ('Rogue_Agent_Autonomy', 'Control_Loss_Probability', +1, 'H', 0),
        ('Rogue_Agent_Autonomy', 'Escalation_Risk', +1, 'H', 0),
    ]
    for src, tgt, sign, conf, delay in edges:
        G.add_edge(src, tgt, sign=sign, confidence=conf, delay=delay)
    return G

def build_governance_graph():
    """19 nodes, 38 edges"""
    G = nx.DiGraph()
    nodes = [
        'Regulatory_Capacity', 'Technical_Expertise_Gov', 'Policy_Response_Speed',
        'International_Coordination', 'Industry_Lobbying_Pressure', 'Public_Demand_Regulation',
        'Regulatory_Capture_Risk', 'Standards_Fragmentation', 'Enforcement_Effectiveness',
        'Innovation_Speed', 'Governance_Gap', 'Institutional_Trust_Gov',
        'Information_Asymmetry', 'Democratic_Accountability', 'Expert_Consensus_Level',
        'Cross_Jurisdictional_Gaps', 'Safety_Standard_Stringency', 'Compliance_Cost',
        'Whistleblower_Protection'
    ]
    G.add_nodes_from(nodes)
    edges = [
        ('Regulatory_Capacity', 'Enforcement_Effectiveness', +1, 'H', 0),
        ('Regulatory_Capacity', 'Policy_Response_Speed', +1, 'H', 1),
        ('Technical_Expertise_Gov', 'Regulatory_Capacity', +1, 'H', 4),
        ('Technical_Expertise_Gov', 'Information_Asymmetry', -1, 'H', 0),
        ('Policy_Response_Speed', 'Governance_Gap', -1, 'H', 1),
        ('Policy_Response_Speed', 'Safety_Standard_Stringency', +1, 'M', 4),
        ('International_Coordination', 'Cross_Jurisdictional_Gaps', -1, 'H', 4),
        ('International_Coordination', 'Standards_Fragmentation', -1, 'M', 4),
        ('Industry_Lobbying_Pressure', 'Regulatory_Capture_Risk', +1, 'H', 1),
        ('Industry_Lobbying_Pressure', 'Safety_Standard_Stringency', -1, 'M', 4),
        ('Public_Demand_Regulation', 'Policy_Response_Speed', +1, 'M', 4),
        ('Public_Demand_Regulation', 'Democratic_Accountability', +1, 'M', 4),
        ('Regulatory_Capture_Risk', 'Enforcement_Effectiveness', -1, 'H', 1),
        ('Regulatory_Capture_Risk', 'Institutional_Trust_Gov', -1, 'M', 4),
        ('Standards_Fragmentation', 'Enforcement_Effectiveness', -1, 'M', 1),
        ('Standards_Fragmentation', 'Compliance_Cost', +1, 'M', 0),
        ('Enforcement_Effectiveness', 'Governance_Gap', -1, 'H', 1),
        ('Enforcement_Effectiveness', 'Innovation_Speed', -1, 'L', 4),
        ('Innovation_Speed', 'Governance_Gap', +1, 'H', 1),
        ('Innovation_Speed', 'Information_Asymmetry', +1, 'M', 0),
        ('Governance_Gap', 'Public_Demand_Regulation', +1, 'M', 4),
        ('Governance_Gap', 'Institutional_Trust_Gov', -1, 'H', 4),
        ('Institutional_Trust_Gov', 'Public_Demand_Regulation', +1, 'M', 4),
        ('Institutional_Trust_Gov', 'Democratic_Accountability', +1, 'M', 4),
        ('Institutional_Trust_Gov', 'Regulatory_Capacity', +1, 'M', 4),
        ('Information_Asymmetry', 'Regulatory_Capacity', -1, 'M', 0),
        ('Information_Asymmetry', 'Regulatory_Capture_Risk', +1, 'M', 1),
        ('Democratic_Accountability', 'Regulatory_Capture_Risk', -1, 'M', 4),
        ('Democratic_Accountability', 'Whistleblower_Protection', +1, 'L', 4),
        ('Expert_Consensus_Level', 'Policy_Response_Speed', +1, 'M', 4),
        ('Expert_Consensus_Level', 'Technical_Expertise_Gov', +1, 'M', 4),
        ('Cross_Jurisdictional_Gaps', 'Governance_Gap', +1, 'H', 0),
        ('Safety_Standard_Stringency', 'Compliance_Cost', +1, 'H', 0),
        ('Safety_Standard_Stringency', 'Enforcement_Effectiveness', +1, 'M', 1),
        ('Compliance_Cost', 'Industry_Lobbying_Pressure', +1, 'M', 1),
        ('Compliance_Cost', 'Innovation_Speed', -1, 'L', 4),
        ('Whistleblower_Protection', 'Information_Asymmetry', -1, 'M', 4),
        ('Whistleblower_Protection', 'Regulatory_Capacity', +1, 'L', 4),
    ]
    for src, tgt, sign, conf, delay in edges:
        G.add_edge(src, tgt, sign=sign, confidence=conf, delay=delay)
    return G

def build_supergraph(cyber, cbrn, deception, autonomy, governance):
    """Build cross-domain convergence supergraph with shared intermediaries and endpoints."""
    G = nx.DiGraph()

    # Add all domain nodes with domain labels
    for name, graph in [('cyber', cyber), ('cbrn', cbrn), ('deception', deception),
                        ('autonomy', autonomy), ('governance', governance)]:
        for n in graph.nodes():
            G.add_node(n, domain=name)
        for u, v, d in graph.edges(data=True):
            G.add_edge(u, v, **d, domain=name, cross_domain=False)

    # Shared intermediary nodes
    shared = ['Trust_Erosion', 'Institutional_Overload', 'Detection_Degradation',
              'Grievance_Amplification', 'Response_Delay_Shared']
    for s in shared:
        G.add_node(s, domain='shared')

    # Catastrophic endpoints
    endpoints = ['Mass_Casualty_Opportunity', 'Critical_Infrastructure_Destabilization',
                 'Democratic_Process_Subversion', 'Catastrophic_Public_Harm',
                 'Irreversible_Loss_of_Control']
    for e in endpoints:
        G.add_node(e, domain='endpoint')

    # Cross-domain edges: domain nodes -> shared intermediaries
    cross_edges = [
        # Cyber -> shared
        ('Incident_Burden', 'Trust_Erosion', +1, 'M'),
        ('Incident_Burden', 'Institutional_Overload', +1, 'H'),
        ('Response_Delay', 'Response_Delay_Shared', +1, 'H'),
        ('Detection_Evasion', 'Detection_Degradation', +1, 'M'),
        # CBRN -> shared
        ('Misuse_Opportunity', 'Trust_Erosion', +1, 'M'),
        ('Misuse_Opportunity', 'Institutional_Overload', +1, 'M'),
        ('Dangerous_Query_Volume', 'Institutional_Overload', +1, 'L'),
        ('Grievance_Level', 'Grievance_Amplification', +1, 'M'),
        ('Oversight_Delay', 'Response_Delay_Shared', +1, 'M'),
        # Deception -> shared
        ('Disinformation_Volume', 'Trust_Erosion', +1, 'H'),
        ('Disinformation_Volume', 'Detection_Degradation', +1, 'H'),
        ('Disinformation_Volume', 'Institutional_Overload', +1, 'M'),
        ('Grievance_Amplification', 'Grievance_Amplification', +1, 'H'),  # self-amplifying
        ('Polarization_Level', 'Grievance_Amplification', +1, 'H'),
        # Autonomy -> shared
        ('Control_Loss_Probability', 'Trust_Erosion', +1, 'M'),
        ('Escalation_Risk', 'Institutional_Overload', +1, 'M'),
        ('Incident_Severity_Ceiling', 'Institutional_Overload', +1, 'L'),
        ('Rogue_Agent_Autonomy', 'Trust_Erosion', +1, 'M'),
        ('Rogue_Agent_Autonomy', 'Institutional_Overload', +1, 'M'),
        # Governance -> shared
        ('Governance_Gap', 'Institutional_Overload', +1, 'H'),
        ('Governance_Gap', 'Response_Delay_Shared', +1, 'H'),
        ('Institutional_Trust_Gov', 'Trust_Erosion', -1, 'M'),
        ('Information_Asymmetry', 'Detection_Degradation', +1, 'M'),
        # Shared -> domain nodes (feedback)
        ('Trust_Erosion', 'Public_Trust_Cyber', -1, 'M'),
        ('Trust_Erosion', 'Epistemic_Trust', -1, 'H'),
        ('Trust_Erosion', 'Institutional_Trust_Gov', -1, 'M'),
        ('Institutional_Overload', 'Defender_Capacity', -1, 'M'),
        ('Institutional_Overload', 'Review_Capacity', -1, 'M'),
        ('Institutional_Overload', 'Regulatory_Capacity', -1, 'M'),
        ('Detection_Degradation', 'Monitoring_Effectiveness', -1, 'M'),
        ('Detection_Degradation', 'Red_Team_Detection_Sensitivity', -1, 'L'),
        ('Grievance_Amplification', 'Grievance_Level', +1, 'M'),
        ('Response_Delay_Shared', 'Response_Delay', +1, 'M'),
        ('Response_Delay_Shared', 'Oversight_Delay', +1, 'M'),
        ('Response_Delay_Shared', 'Policy_Response_Speed', -1, 'M'),
    ]
    for item in cross_edges:
        src, tgt, sign, conf = item
        if G.has_node(src) and G.has_node(tgt):
            G.add_edge(src, tgt, sign=sign, confidence=conf, delay=0,
                      cross_domain=True, domain='cross')

    # Shared intermediaries -> endpoints (indirect social/institutional pathways)
    endpoint_edges = [
        ('Trust_Erosion', 'Democratic_Process_Subversion', +1, 'M'),
        ('Trust_Erosion', 'Catastrophic_Public_Harm', +1, 'M'),
        ('Institutional_Overload', 'Critical_Infrastructure_Destabilization', +1, 'H'),
        ('Institutional_Overload', 'Mass_Casualty_Opportunity', +1, 'M'),
        ('Institutional_Overload', 'Catastrophic_Public_Harm', +1, 'H'),
        ('Detection_Degradation', 'Critical_Infrastructure_Destabilization', +1, 'M'),
        ('Detection_Degradation', 'Mass_Casualty_Opportunity', +1, 'M'),
        ('Grievance_Amplification', 'Mass_Casualty_Opportunity', +1, 'L'),
        ('Grievance_Amplification', 'Democratic_Process_Subversion', +1, 'M'),
        ('Response_Delay_Shared', 'Critical_Infrastructure_Destabilization', +1, 'H'),
        ('Response_Delay_Shared', 'Mass_Casualty_Opportunity', +1, 'M'),
        ('Response_Delay_Shared', 'Catastrophic_Public_Harm', +1, 'M'),
        ('Response_Delay_Shared', 'Irreversible_Loss_of_Control', +1, 'M'),
        ('Institutional_Overload', 'Irreversible_Loss_of_Control', +1, 'M'),
        ('Trust_Erosion', 'Irreversible_Loss_of_Control', +1, 'L'),
    ]
    for src, tgt, sign, conf in endpoint_edges:
        G.add_edge(src, tgt, sign=sign, confidence=conf, delay=0,
                  cross_domain=True, domain='endpoint_link')

    # Direct domain -> endpoint edges (direct technical pathways)
    direct_endpoint_edges = [
        ('Active_Exploits', 'Critical_Infrastructure_Destabilization', +1, 'H'),
        ('Misuse_Opportunity', 'Mass_Casualty_Opportunity', +1, 'H'),
        ('Election_Integrity', 'Democratic_Process_Subversion', -1, 'H'),
        ('Escalation_Risk', 'Catastrophic_Public_Harm', +1, 'M'),
        ('Control_Loss_Probability', 'Catastrophic_Public_Harm', +1, 'M'),
        ('Rogue_Agent_Autonomy', 'Irreversible_Loss_of_Control', +1, 'H'),
        ('Override_Latency', 'Irreversible_Loss_of_Control', +1, 'M'),
        ('Interpretability_Gap', 'Irreversible_Loss_of_Control', +1, 'M'),
        ('Governance_Gap', 'Catastrophic_Public_Harm', +1, 'M'),
    ]
    for src, tgt, sign, conf in direct_endpoint_edges:
        if G.has_node(src):
            G.add_edge(src, tgt, sign=sign, confidence=conf, delay=0,
                      cross_domain=False, domain='direct_endpoint')

    return G, shared, endpoints

def get_all_graphs():
    """Build and return all graphs."""
    cyber = build_cyber_graph()
    cbrn = build_cbrn_graph()
    deception = build_deception_graph()
    autonomy = build_autonomy_graph()
    governance = build_governance_graph()
    supergraph, shared_nodes, endpoint_nodes = build_supergraph(
        cyber, cbrn, deception, autonomy, governance)
    return {
        'cyber': cyber, 'cbrn': cbrn, 'deception': deception,
        'autonomy': autonomy, 'governance': governance,
        'supergraph': supergraph, 'shared_nodes': shared_nodes,
        'endpoint_nodes': endpoint_nodes
    }
