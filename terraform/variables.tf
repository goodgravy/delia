# ----------------------------------------#
#
#       | Terraform Variables file |
#
# ----------------------------------------#
# File: variables.tf
# Author: Vithursan Thangarasa (vithursant)
# ----------------------------------------#

variable "my_region" {
  type    = string
  default = "eu-west-1"
  description = "The AWS region to deploy into (i.e. us-east-1)"
}

variable "avail_zone" {
  type    = string
  default = "eu-west-1a"
  description = "The AWS availability zone location within the selected region (i.e. us-east-2a)."
}

variable "my_ip" {
  type    = string
  default = "0.0.0.0/0"
  #default = "185.41.96.0/24"
}

variable "my_cidr_block" {
  type    = string
  default = "10.0.0.0/24"
}

variable "my_key_pair_name" {
  type    = string
  default = "james-development-eu-west-1"
  description = "The name of the SSH key to install onto the instances."
}

variable "ssh-key-dir" {
  default     = "~/.ssh/"
  description = "The path to SSH keys - include ending '/'"
}

variable "instance_type" {
  type    = string
  default = "p2.xlarge"
  description = "The instance type to provision the instances from (i.e. p2.xlarge)."

}

variable "spot_price" {
  type    = string
  default = "0.50"
  description = "The maximum hourly price (bid) you are willing to pay for the specified instance, i.e. 0.10. This price should not be below AWS' minimum spot price for the instance based on the region."
}

variable "ebs_volume_id" {
  type = string
  default = "vol-050aa74a11a5cf1db"
  description = "The EBS volume that you would like to attach to the instance. This volume should already exist: it will not be managed by Terraform"
}

variable "ami_id" {
  type    = string
  default = "ami-003c19688b5ed7925" # Default AWS Deep Learning AMI (Ubuntu)
  description = "The AMI ID to use for each instance. The AMI ID will be different depending on the region, even though the name is the same."
}

variable "num_instances" {
  type    = string
  default = "1"
  description = "The number of AWS EC2 instances to provision."
}
