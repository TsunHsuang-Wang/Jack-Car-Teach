from jack_car import JackCar

def main1():
    DP_method = 'value-iteration'
    p_dict = {
        'conf_path': './env_dynamics/a1_conf.npy',
        'P_path': './env_dynamics/a1_P.npy',
        'R_path': './env_dynamics/a1_R.npy'
    }
    model = JackCar(True, precomputed_dict=p_dict)
    
    print('JackCar model:')
    print('    maximum move: {}'.format(model.max_move))
    print('    maximum number of cars in A: {}'.format(model.max_cars_A))
    print('    maximum number of cars in B: {}'.format(model.max_cars_B))
    print('    rent price: {}'.format(model.rent_price))
    print('    move cost: {}'.format(model.move_cost))
    print('    discount factor of MDP: {}'.format(model.gamma))

    model.run(DP_method)
    model.visualize_policy()

def main2():
    save_file = True
    save_file_name = './env_dynamics/a3'
    DP_method = 'policy-iteration'
    conf = {
        'max_move': 4,
        'max_cars_A': 7,
        'max_cars_B': 10,
        'rent_price': 10,
        'move_cost': 2,
        'gamma': 0.9
    }
    model = JackCar(False, conf=conf)
    
    print('JackCar model:')
    print('    maximum move: {}'.format(model.max_move))
    print('    maximum number of cars in A: {}'.format(model.max_cars_A))
    print('    maximum number of cars in B: {}'.format(model.max_cars_B))
    print('    rent price: {}'.format(model.rent_price))
    print('    move cost: {}'.format(model.move_cost))
    print('    discount factor of MDP: {}'.format(model.gamma))

    if save_file:
        model.save_full_env_dynamics(save_file_name)
    
    model.run()
    model.visualize_policy()

if __name__=='__main__':
    main1()

