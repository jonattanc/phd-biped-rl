# test_fixed.py
import os
import sys
import numpy as np
import time
import pybullet as p

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import Simulation
from robot import Robot
from environment import Environment
import utils

class MockValue:
    def __init__(self, initial=0):
        self.value = initial

def test_robot_initialization():
    """Testa se o robô é inicializado corretamente"""
    print("=== TESTE DE INICIALIZAÇÃO DO ROBÔ ===")
    
    try:
        robot = Robot("robot_stage1")
        env_obj = Environment("PR")
        
        # Verificar se os arquivos URDF foram gerados
        assert os.path.exists(robot.urdf_path), f"URDF do robô não encontrado: {robot.urdf_path}"
        assert os.path.exists(env_obj.urdf_path), f"URDF do ambiente não encontrado: {env_obj.urdf_path}"
        
        print("✅ URDFs gerados corretamente")
        return True
        
    except Exception as e:
        print(f"❌ Erro na inicialização: {e}")
        return False

def test_simulation_setup():
    """Testa a configuração básica da simulação"""
    print("\n=== TESTE DE CONFIGURAÇÃO DA SIMULAÇÃO ===")
    
    try:
        robot = Robot("robot_stage1")
        env_obj = Environment("PR")
        sim_env = Simulation(robot, env_obj, MockValue(0), MockValue(0), MockValue(0), enable_gui=False)
        
        # Verificar se a simulação foi configurada
        assert sim_env.physics_client is not None, "Cliente de física não inicializado"
        assert sim_env.robot.id is not None, "Robô não carregado na simulação"
        assert sim_env.environment.id is not None, "Ambiente não carregado"
        
        print("✅ Simulação configurada corretamente")
        
        # Verificar espaços de ação e observação
        print(f"✅ Action space: {sim_env.action_space}")
        print(f"✅ Observation space: {sim_env.observation_space}")
        
        sim_env.close()
        return True
        
    except Exception as e:
        print(f"❌ Erro na configuração da simulação: {e}")
        return False

def test_improved_reset():
    """Testa o reset melhorado"""
    print("\n=== TESTE DE RESET MELHORADO ===")
    
    try:
        robot = Robot("robot_stage1")
        env_obj = Environment("PR")
        sim_env = Simulation(robot, env_obj, MockValue(0), MockValue(0), MockValue(0), enable_gui=False)
        
        for i in range(2):
            print(f"\n--- Reset {i+1} ---")
            obs, info = sim_env.reset()
            
            # Verificar observação
            print(f"✅ Observação shape: {obs.shape}")
            print(f"✅ Observação valores: {np.array2string(obs, precision=3, suppress_small=True)}")
            
            # Verificar posição do robô
            pos, orient = p.getBasePositionAndOrientation(sim_env.robot.id)
            print(f"✅ Posição inicial: z={pos[2]:.3f}")
            print(f"✅ Fall threshold: {sim_env.fall_threshold}")
            
            # Verificar estabilidade
            if pos[2] > sim_env.fall_threshold + 0.1:  # Margem de segurança
                print("✅ Robô está estável")
            else:
                print(f"⚠️  Robô pode instável (z={pos[2]:.3f})")
        
        sim_env.close()
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste de reset: {e}")
        return False

def test_conservative_step():
    """Testa steps com parâmetros conservadores"""
    print("\n=== TESTE DE STEP CONSERVADOR ===")
    
    try:
        robot = Robot("robot_stage1")
        env_obj = Environment("PR")
        sim_env = Simulation(robot, env_obj, MockValue(0), MockValue(0), MockValue(0), enable_gui=False)
        
        obs, _ = sim_env.reset()
        episode_data = []
        
        # Testar com ações pequenas
        for step in range(50):
            # Ação conservadora (pequena amplitude)
            action = np.random.uniform(-0.2, 0.2, size=sim_env.action_space.shape)
            obs, reward, terminated, truncated, info = sim_env.step(action)
            
            episode_data.append({
                'step': step,
                'reward': reward,
                'distance': info['distance'],
                'terminated': terminated
            })
            
            if step % 10 == 0:
                pos, _ = p.getBasePositionAndOrientation(sim_env.robot.id)
                print(f"  Step {step}: reward={reward:7.2f}, dist={info['distance']:6.3f}, z={pos[2]:5.3f}")
            
            if terminated or truncated:
                print(f"Episódio terminou no step {step}: {info.get('termination', 'unknown')}")
                break
        else:
            print("✅ Episódio completou 50 steps!")
        
        # Análise
        total_steps = len(episode_data)
        final_reward = sum(d['reward'] for d in episode_data)
        
        print(f"\n📊 RESUMO:")
        print(f"   Total de steps: {total_steps}")
        print(f"   Recompensa total: {final_reward:.2f}")
        
        # Critério de sucesso: pelo menos 10 steps
        success = total_steps >= 10
        print("✅ TESTE PASSOU" if success else "⚠️  TESTE FALHOU")
        
        sim_env.close()
        return success
        
    except Exception as e:
        print(f"❌ Erro no teste de step: {e}")
        return False

def main():
    """Executa todos os testes corrigidos"""
    print("🧪 INICIANDO TESTES CORRIGIDOS")
    print("=" * 60)
    
    # Garantir diretórios
    os.makedirs(utils.TMP_PATH, exist_ok=True)
    os.makedirs(utils.LOGS_PATH, exist_ok=True)
    
    # Testes em ordem de complexidade
    tests = [
        test_robot_initialization,
        test_simulation_setup,
        test_improved_reset,
        test_conservative_step,
    ]
    
    results = []
    for test in tests:
        try:
            print(f"\n{'='*50}")
            result = test()
            results.append(result)
            print(f"{'='*50}")
        except Exception as e:
            print(f"❌ Teste {test.__name__} falhou: {e}")
            results.append(False)
    
    # Resumo
    print("\n🎯 RESUMO FINAL")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Testes passados: {passed}/{total}")
    
    if passed == total:
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("\n📝 PRÓXIMOS PASSOS RECOMENDADOS:")
        print("1. Execute a GUI para treinamento completo")
        print("2. Monitore se o robô mantém o equilíbrio")
        print("3. Ajuste gradualmente os parâmetros se necessário")
    else:
        print(f"⚠️  {total - passed} teste(s) falhou")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)