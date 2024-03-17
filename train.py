def main() -> None:
    # Load and preprocess the dataset
    train_dataset, val_dataset, test_dataset = load_data(config["data"]["local_dir"])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"])

    # Create the model, optimizer, and loss function
    model, optimizer, loss_fn = create_model()

    # Train the model
    metrics = train(model, optimizer, loss_fn, train_loader, val_loader)
    logger.info(f"Validation metrics: {metrics}")

    # Save artifacts
    os.makedirs(config["artifacts"]["output_dir"], exist_ok=True)
    if config["artifacts"]["save_best_model"]:
        torch.save(model.state_dict(), os.path.join(config["artifacts"]["output_dir"], "best_model.pth"))
    if config["artifacts"]["save_logs"]:
        # Save logs
        with open(os.path.join(config["artifacts"]["output_dir"], "training.log"), "w") as log_file:
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    log_file.write(handler.stream.getvalue())
    if config["artifacts"]["save_metrics"]:
        # Save metrics
        with open(os.path.join(config["artifacts"]["output_dir"], "metrics.json"), "w") as metrics_file:
            json.dump(metrics, metrics_file)
