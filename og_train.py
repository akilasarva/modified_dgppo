import argparse
import datetime
import os
import ipdb
import numpy as np
import wandb
import yaml

from dgppo.algo import make_algo
from dgppo.env import make_env
from dgppo.trainer.trainer import Trainer
from dgppo.trainer.utils import is_connected


def train(args):
    print(f"> Running train.py {args}")

    # set up environment variables and seed
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if not is_connected():
        os.environ["WANDB_MODE"] = "offline"
    np.random.seed(args.seed)
    if args.debug:
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["JAX_DISABLE_JIT"] = "True"

    # create environments
    env = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        n_rays=args.n_rays,
        full_observation=args.full_observation,
    )
    env_test = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        n_rays=args.n_rays,
        full_observation=args.full_observation,
    )

    # create algorithm
    algo = make_algo(
        algo=args.algo,
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        cost_weight=args.cost_weight,
        cbf_weight=args.cbf_weight,
        actor_gnn_layers=args.actor_gnn_layers,
        Vl_gnn_layers=args.Vl_gnn_layers,
        Vh_gnn_layers=args.Vh_gnn_layers,
        rnn_layers=args.rnn_layers,
        lr_actor=args.lr_actor,
        lr_Vl=args.lr_Vl,
        lr_Vh=args.lr_Vh,
        max_grad_norm=2.0,
        alpha=args.alpha,
        cbf_eps=args.cbf_eps,
        seed=args.seed,
        batch_size=args.batch_size,
        use_rnn=not args.no_rnn,
        use_lstm=args.use_lstm,
        coef_ent=args.coef_ent,
        rnn_step=args.rnn_step,
        gamma=0.99,
        clip_eps=args.clip_eps,
        lagr_init=args.lagr_init,
        lr_lagr=args.lr_lagr,
        train_steps=args.steps,
        cbf_schedule=not args.no_cbf_schedule,
        cost_schedule=args.cost_schedule
    )

    # Generate a 4 letter random identifier for the run.
    rng_ = np.random.default_rng()
    rand_id = "".join([chr(rng_.integers(65, 91)) for _ in range(4)])

    # set up logger
    start_time = datetime.datetime.now()
    start_time = start_time.strftime("%m%d%H%M%S")
    if not args.debug:
        if not os.path.exists(f"{args.log_dir}/{args.env}/{args.algo}"):
            os.makedirs(f"{args.log_dir}/{args.env}/{args.algo}", exist_ok=True)
    start_time = int(start_time)
    while os.path.exists(f"{args.log_dir}/{args.env}/{args.algo}/seed{args.seed}_{start_time}_{rand_id}"):
        start_time += 1

    log_dir = f"{args.log_dir}/{args.env}/{args.algo}/seed{args.seed}_{start_time}_{rand_id}"
    run_name = "{}_seed{:03}_{}_{}".format(args.algo, args.seed, start_time, rand_id)
    if args.name is not None:
        run_name = "{}_{}_seed{:03}_{}_{}".format(run_name, args.name, args.seed, start_time, rand_id)

    # get training parameters
    train_params = {
        "run_name": run_name,
        "training_steps": args.steps,
        "eval_interval": args.eval_interval,
        "eval_epi": args.eval_epi,
        "save_interval": args.save_interval
    }

    # create trainer
    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        gamma=0.99,
        log_dir=log_dir,
        n_env_train=args.n_env_train,
        n_env_test=args.n_env_test,
        seed=args.seed,
        params=train_params,
        save_log=not args.debug,
    )

    # save config
    wandb.config.update(args)
    wandb.config.update(algo.config, allow_val_change=True)
    if not args.debug:
        with open(f"{log_dir}/config.yaml", "w") as f:
            yaml.dump(args, f)
            yaml.dump(algo.config, f)

    # start training
    trainer.train()


def main():
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("-n", "--num-agents", type=int, required=True)
    parser.add_argument("--algo", type=str, required=True)
    parser.add_argument("--obs", type=int, required=True)

    # custom arguments
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--cost-weight", type=float, default=0.)
    parser.add_argument("--n-rays", type=int, default=32)
    parser.add_argument('--full-observation', action='store_true', default=False)
    parser.add_argument('--clip-eps', type=float, default=0.25)
    parser.add_argument('--lagr-init', type=float, default=0.5)
    parser.add_argument('--lr-lagr', type=float, default=1e-7)
    parser.add_argument("--cbf-weight", type=float, default=1.0)
    parser.add_argument("--cbf-eps", type=float, default=1e-2)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--no-cbf-schedule", action="store_true", default=False)
    parser.add_argument("--cost-schedule", action="store_true", default=False)
    parser.add_argument("--no-rnn", action="store_true", default=False)

    # NN arguments
    parser.add_argument("--actor-gnn-layers", type=int, default=2)
    parser.add_argument("--Vl-gnn-layers", type=int, default=2)
    parser.add_argument("--Vh-gnn-layers", type=int, default=1)
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-Vl", type=float, default=1e-3)
    parser.add_argument("--lr-Vh", type=float, default=1e-3)
    parser.add_argument("--rnn-layers", type=int, default=1)
    parser.add_argument("--use-lstm", action="store_true", default=False)
    parser.add_argument("--coef-ent", type=float, default=1e-2)
    parser.add_argument("--rnn-step", type=int, default=16)

    # default arguments
    parser.add_argument("--n-env-train", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--n-env-test", type=int, default=32)
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-epi", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=50)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
