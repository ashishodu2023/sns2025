def main():
    """Example main entry point using the factory approach."""
    factory = SNSRawPrepSepDNNFactory()

    # 1) Run entire pipeline
    factory.run_pipeline()

    # or do partial steps:
    # beam_df = factory.create_beam_data()
    # dcm_df = factory.create_dcm_data()
    # ...
    # model = factory.create_vae_bilstm_model(...)

if __name__ == "__main__":
    main()